/* 并行原语：CP 三种取 K / V 的办法、EP 五个后端、FSDP 每层的字节与时间。训练核心和后训练的并行一节都只调这里，
   两边看到的是同一组公式。所有函数按“一层、最忙那张卡”算，调用方再乘层数和文档数。 */
const Par = (function(){
const { link } = Common;

const binomRatio = (Z, z, K) => { let r=1; for(let i=0;i<K;i++) r *= Math.max(0, Z-z-i)/(Z-i); return r; };   // C(Z−z,K)/C(Z,K)
const pDest = (Z, E, K) => E<=1 ? 1 : 1-binomRatio(Z, Z/E, K);   // 一个 token 命中某个 rank 的概率（不放回地选 K 个 expert）
const pow2Dom = topo => Math.pow(2, Math.floor(Math.log2(topo.domain)));   // 2 的幂的组能用满的 NVLink 域：HGX 8，NVL72 64
const lk = (hw, topo, n, stride) => { const l=link(hw,topo,n,stride); return Object.assign({ n, kind: n<=1 ? "—" : l.intra ? "NVLink" : "IB" }, l); };
/* 一段线上传输：calls 次调用的固定延迟 + 字节 / 有效带宽 */
const wireT = (bytes, l, eff, calls, g) => (l.n<=1||bytes<=0) ? 0 : (calls||1)*(l.intra?g.alphaNv:g.alphaIb)*1e-6 + bytes/(l.bw*eff);

const EP = {
  standard:{ name:"NCCL all-to-all（torchtitan 默认）", short:"a2a", col:"s5", fused:false },
  ag:      { name:"AllGather + RS（NCCL）", short:"AG-RS", col:"s1", fused:false },
  deepep:  { name:"DeepEP v2 expanded", short:"DeepEP", col:"s2", fused:false },
  distmoe: { name:"torch Dist-MoE", short:"Dist-MoE", col:"s3", fused:true },
  mok:     { name:"Mixture-of-Kittens", short:"MoK", col:"s4", fused:true } };
const CP = {
  ring:      { name:"Ring", col:"s1" },
  allgather: { name:"AllGather", col:"s2" },
  ulysses:   { name:"Ulysses", col:"s3" } };

/* ---------- CP：一个 attention 层、一篇文档 ----------
   x：S 文档长度、C、tp、m、d、hw、topo、g、ac（full 重算）、tCore（这一层这一篇的核心 F+B 时间）、kind：full / swa */
function cpLayer(meth, x){
  const { S, C, tp, m, d, hw, topo, g, ac, tCore, kind } = x, f=m.full;
  const o = { meth, wire:0, calls:0, tw:0, exposed:0, lk:lk(hw,topo,C,tp), ok:true, why:"" };
  if(C<=1) return o;
  const l = o.lk, gqa = f.kind==="gqa";
  const kvTok = (gqa ? d.kvTokElems/Math.min(tp, f.Hkv) : d.kvTokElems) + (f.kind==="dsa" ? f.idxD : 0);   // 每 token 的 K+V（或 latent + indexer key）元素
  const Lb = 2*(S/C)*kvTok;                                     // 本卡一份 K + V 的字节
  if(kind==="swa" && meth!=="ulysses"){                         // 滑窗：只和邻居交换最后 W 个 token
    o.calls = ac?4:3; o.wire = o.calls*2*Math.min(f.W, S/C)*kvTok;
    o.tw = wireT(o.wire, l, g.etaAg, o.calls, g); o.exposed = o.tw; return o;
  }
  if(meth==="allgather"){
    const chunks = gqa ? Math.min(f.Hkv/Math.min(tp,f.Hkv), 2) : 1;
    o.wire = (ac?4:3)*(C-1)*Lb; o.calls = (ac?8:6)*chunks;
    o.tw = wireT(o.wire, l, g.etaAg, o.calls, g);
  } else if(meth==="ring"){
    o.wire = (ac ? 5*C-3 : 4*C-2)*Lb; o.calls = 2*(C-1)+C + (ac?C-1:0);
    o.tw = wireT(o.wire, l, g.etaAg, o.calls, g);
  } else {
    const H = d.Hattn/tp;
    const units = gqa ? 2*H*f.dh + 2*Math.max(f.Hkv/Math.min(tp,f.Hkv), C)*f.dh : H*(d.dqk+d.dv) + C*kvTok;   // MLA 的 latent 不能按头切，按 C 份复制
    o.wire = (ac?3:2)*2*(S/C)*units*(C-1)/C; o.calls = ac?12:8;
    o.tw = wireT(o.wire, l, g.etaN, o.calls, g);
    if(H % C){ o.ok=false; o.why = "Ulysses 要求每个 TP rank 的注意力头数 "+H+" 能被 CP="+C+" 整除"; }
  }
  // Ring / AllGather 可以和分块计算重叠，只露出第一块和盖不住的部分；Ulysses 的两次 all-to-all 全部阻塞
  o.exposed = meth==="ulysses" ? o.tw : o.tw/C + Math.max(0, o.tw - tCore*(C-1)/C);
  return o;
}

/* ---------- EP：一层 MoE 的通信 + expert GEMM（F+B，含重算） ----------
   x：n 本卡 token、K、Z、E、De、stride、hw、topo、g、ac、prec、imb、tExpF / tExpB（这一层 expert 计算的前向 / 反向含重算）、syncUs、mokF */
function epLayer(be, x){
  const { n, K, Z, E, De, hw, topo, g, ac, imb } = x, tExpF=x.tExpF||0, tExpB=x.tExpB||0, tExp=tExpF+tExpB;
  const o = { be, ok:true, why:"", wire:0, ib:0, tw:0, tAlpha:0, tSync:0, t:tExp, extra:0, exch:0, syncs:0, buf:0, lk:lk(hw,topo,E,x.stride||1), parts:null, det:be!=="deepep" };
  if(E<=1){ o.parts = { f:[["sm","expert GEMM",tExpF,"moe"]], b:[["sm","expert dgrad + wgrad",tExpB,"moe"]] }; o.buf = n*K*imb*De*2; return o; }
  const l = o.lk, mx = /^mx|fp8/.test(x.prec||"");
  const bd = (mx && (be==="deepep"||be==="distmoe"||be==="mok")) ? 1+1/32 : 2;          // MXFP8 量化融合进 dispatch
  const slots = n*K*imb*(1-1/E), dE = (E-1)*pDest(Z,E,K);
  let disp, comb;
  if(be==="ag"){ disp=(E-1)*n*De*bd; comb=(E-1)*n*De*2; }
  else if(be==="deepep"){ disp=n*imb*dE*De*bd; comb=n*imb*dE*De*2; }
  else { disp=slots*De*bd; comb=slots*De*2; }
  const dispB = disp/bd*2, combB = comb;
  const replay = ac && (be==="standard"||be==="ag");                                      // 重算时 NCCL 路径要重放 dispatch / combine
  o.exch = 4 + (replay?2:0);
  o.wire = disp + comb + dispB + combB + (replay ? disp+comb : 0);
  const effN = be==="ag" ? g.etaAg : g.etaN, dom = pow2Dom(topo);
  /* 线上时间：跨域时 NCCL a2a 按均匀路由的远端比例；DeepEP 每个远端节点只发一次再 NVLink 扇出；AG 环全程按最慢链路 */
  const tWire = w => {
    if(l.intra) return w/(hw.nvl*1e9*effN);
    if(be==="ag"){ o.ib += w; return w/(hw.ib*1e9*g.etaAg); }
    if(be==="deepep"){ const nodes=E/dom, rn=(nodes-1)*(1-binomRatio(Z, Z/nodes, K)); const ibw = w*rn/Math.max(dE,1e-9); o.ib += ibw; return Math.max(ibw/(hw.ib*1e9*g.etaN), w/(hw.nvl*1e9*effN)); }
    const rf=(E-dom)/(E-1); o.ib += w*rf; return Math.max(w*rf/(hw.ib*1e9*g.etaN), w*(1-rf)/(hw.nvl*1e9*effN));
  };
  const a = (l.intra ? g.alphaNv : g.alphaIb)*1e-6;
  const callsPer = be==="standard"||be==="ag" ? 2 : be==="deepep" ? 1 : 0;
  const aOne = be==="deepep" ? g.deepepUs*1e-6 + (l.intra?0:g.alphaIb*1e-6) : a;
  o.tAlpha = o.exch*callsPer*aOne + (be==="distmoe" ? 4*g.barrierUs*1e-6 + 2e-5 : 0);
  o.syncs = be==="standard" ? (replay?2:1) : be==="deepep" ? 1 : 0;
  o.tSync = o.syncs*(x.syncUs||0)*1e-6;
  const tD = tWire(disp), tC = tWire(comb), tDB = tWire(dispB), tCB = tWire(combB);
  o.tw = tD + tC + tDB + tCB + (replay ? tD + tC : 0);
  const al = o.tAlpha/o.exch||0, lane = l.intra ? "nvl" : "ib";
  if(!EP[be].fused){
    o.t = tExp + o.tw + o.tAlpha + o.tSync;
    const nm = be==="ag" ? ["all-gather token","reduce-scatter 输出"] : be==="deepep" ? ["dispatch（按 rank 去重）","combine"] : ["all-to-all dispatch","all-to-all combine"];
    const f = [];
    if(o.syncs) f.push(["cpu", be==="deepep"?"do_cpu_sync：CPU 轮询计数":"D2H：拷 output_splits", o.tSync/o.syncs, "sync"]);
    f.push([lane, nm[0], tD+al, "ep"], ["sm","expert GEMM",tExpF,"moe"], [lane, nm[1], tC+al, "ep"]);
    const b = [];
    if(replay) b.push([lane,"重算：重放 dispatch",tD+al,"ep"], [lane,"重算：重放 combine",tC+al,"ep"]);
    if(replay && o.syncs>1) b.unshift(["cpu","重放时再拷一次 split",o.tSync/o.syncs,"sync"]);
    b.push([lane,"combine 的反向",tCB+al,"ep"], ["sm","expert dgrad + wgrad"+(ac?"（含重算）":""),tExpB,"moe"], [lane,"dispatch 的反向",tDB+al,"ep"]);
    o.parts = { f, b };
  } else if(be==="distmoe"){
    o.t = Math.max(tExp/0.92, o.tw) + o.tAlpha;
    const sF = Math.max(tExpF/0.92, tD+tC), sB = Math.max(tExpB/0.92, tDB+tCB), bar = g.barrierUs*1e-6;
    o.parts = { f:[["sm","barrier + 路由指针表",bar+1e-5,"sync"],["sm","W13 gather warp 拉对端行 + GEMM → W2 epilogue 写对端 combine",sF,"moe"],["nvl","对端 load / store（融合在 GEMM 里）",tD+tC,"ep",true],["sm","barrier + 本地 K 路 scale_and_sum",bar,"sync"]],
                b:[["sm","barrier + 路由",bar+1e-5,"sync"],["sm","反向融合 GEMM（dgrad 拉 / wgrad）",sB,"moe"],["nvl","对端 load / store",tDB+tCB,"ep",true],["sm","barrier + 归约",bar,"sync"]] };
    if(!l.intra){ o.ok=false; o.why="Dist-MoE 用 symmetric memory 的指针 load / store，EP 组只能在一个 NVLink 域内"; }
    if(hw.sm<100){ o.ok=false; o.why="Dist-MoE 的融合 kernel 只支持 SM100+"; }
  } else {
    const fr = Math.min(0.6, Math.max(0.05, x.mokF||0.25)), wEff = Math.min(1, fr/0.3);   // 通信 SM 比例 f：f ≥ 0.3 时打满 NVLink
    o.t = Math.max(tExp/(1-fr), o.tw/wEff) + 0.03*tExp;
    const sF = Math.max(tExpF/(1-fr), (tD+tC)/wEff), sB = Math.max(tExpB/(1-fr), (tDB+tCB)/wEff);
    o.parts = { f:[["sm","MoK megakernel：计算 SM 跑 grouped GEMM + SwiGLU",sF,"moe"],["nvl","通信 SM：pull dispatch / push combine",(tD+tC)/wEff,"ep",true]],
                b:[["sm","MoK 反向：reverse-combine → dgrad → reverse-dispatch，wgrad 最后整轴累加",sB,"moe"],["nvl","通信 SM",(tDB+tCB)/wEff,"ep",true]] };
    if(topo.domain<64 || !/^GB/.test(hw.name)){ o.ok=false; o.why="MoK 面向 GB200 / GB300 NVL72（单 NVLink 域）"; }
    else if(!l.intra){ o.ok=false; o.why="MoK 的 EP 组必须在一个 NVL72 机柜内"; }
  }
  o.extra = o.t - tExp;                                            // 关键路径上比纯 GEMM 多出来的部分（含 host sync）
  o.buf = be==="standard" ? 2*n*K*imb*De*2 : be==="ag" ? 2*E*n*De*2 : be==="deepep" ? n*K*imb*De*2 + 1e9 : be==="distmoe" ? 2*n*K*De*2 + n*K*imb*De*2 : 0.4e9;
  return o;
}

/* ---------- FSDP：一层参数 P 在 F 张卡上切分 ---------- */
const fsdpBytes = (P, F, aG, gB) => F>1 ? P*(F-1)/F*(aG*2+gB) : 0;
const fsdpT = (P, F, l, g, aG, gB) => wireT(fsdpBytes(P,F,aG,gB), l, g.etaAg, aG+1, g);

return { EP, CP, pDest, binomRatio, pow2Dom, lk, wireT, cpLayer, epLayer, fsdpBytes, fsdpT };
})();
