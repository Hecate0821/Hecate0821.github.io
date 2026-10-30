/* ================= 低精度：推理、训练各一节；格式放附录，RL 组合放 RL 页 ================= */
const QC = { bf16:css("c-lm"), fp8:css("c-attn"), mxfp8:css("c-lin"), w4a16:css("c-ar"), w4a8:css("c-cp"), mxfp4:css("c-moe"), nvfp4:css("c-a2a") };
const QE2E = { bf16:{wDt:"bf16",eAct:"",dDt:"bf16"}, fp8:{wDt:"fp8",eAct:"",dDt:"fp8"}, mxfp8:{wDt:"mxfp8",eAct:"",dDt:"mxfp8"},
  w4a16:{wDt:"mxfp4",eAct:"bf16",dDt:"bf16"}, w4a8:{wDt:"mxfp4",eAct:"mxfp8",dDt:"mxfp8"}, mxfp4:{wDt:"mxfp4",eAct:"",dDt:"bf16"}, nvfp4:{wDt:"nvfp4",eAct:"",dDt:"bf16"} };
const QTRN = ["bf16","fp8","mxfp8","nvfp4"];
const DTN = { bf16:"BF16", fp8:"FP8 block", mxfp8:"MXFP8", mxfp4:"MXFP4", nvfp4:"NVFP4" };
const qOpt = (over) => Object.assign({ fuse:S.q.fuse, align:1, floor:false, BM:S.q.BM, deqRate:S.q.deqRate }, over||{});   // 单个 GEMM 按纯 roofline，不计 kernel 最短时间
const qTopo = h => /^GB/.test(h) ? "nvl64" : (S.topo==="nvl64" ? "hgx8" : S.topo);
function qShape(){ const m=S.model, e=m.moe, De=e.Dlat||m.D, F=e.F||m.dense.F;   // dense 模型：gate_up / down 取 dense FFN
  const sh={ up:[2*F,De], down:[De,F], shared:[2*(e.Fs||m.dense.F||e.F),m.D] }[S.q.shape]; if(sh){ S.q.N=sh[0]; S.q.K=sh[1]; } }
const fX = x => (x>=10?x.toFixed(1):x.toFixed(2))+"×";
const bits = d => (Common.dtB(d)*8).toFixed(d==="fp8"?2:2);

function qInferE2E(hw, key){ const {m,g}=ctx(), topo=TOPO[qTopo(S.q.hw)];
  const c=Object.assign({},S.inf,{phase:"decode",qSep:S.q.fuse==="separate"},QE2E[key]);
  if(!m.moe.E){ c.dDt=c.wDt; c.dAct=c.eAct; }     // dense 模型没有 expert：方案作用于全部线性层
  return Infer.decode(m,hw,topo,c,g); }
function qTrainE2E(hw, prec){ const {m,g}=ctx(), topo=TOPO[qTopo(S.q.hw)];
  return Train.run(m,hw,topo,Object.assign({N:topo.gpus},S.trn,{prec,tokFused:S.q.tokFused}),g); }

/* 每条实测在这一页的公式下算出来的对应数字 */
function qCheck(key, hw){
  const g=S.glob, Q=Quant, o=qOpt();
  if(key==="peaks"){ const B=HW.B200, B3=HW.B300; return "HW 表：B200 "+B.bf16/1000+" / "+B.fp8/1000+" / "+B.fp4/1000+"，B300 FP4 "+B3.fp4/1000+" PF ✓"; }
  if(key==="castBW"){ const bw=HW.B200.hbm*g.etaM/1000; return "η<sub>m</sub> = "+g.etaM+" → "+bw.toFixed(1)+" TB/s；同形状按 3 B/元素算 "+fT(16384*16384*(2+1+1/32)/(bw*1e12))+" ✓"; }
  if(key==="w4a8"){ const sh=[[4096,6144],[6144,2048]], r=[];
    for(const [N,K] of sh) for(const M of [256,4096]){ const a=Q.gemm(HW.B200,g,"w4a16",M,N,K,o), b=Q.gemm(HW.B200,g,"w4a8",M,N,K,o); r.push(a.t/b.t); }
    const qw={moe:{E:256,k:8,F:512,Dlat:0},D:2048}, sep=qOpt({fuse:"separate",align:"auto"});
    const e2=[64,1024].map(T=>Q.moeLayer(HW.B200,g,qw,"w4a16",T,1,sep).t/Q.moeLayer(HW.B200,g,qw,"w4a8",T,1,sep).t);
    const lo=fX(Math.min(...r)), hi=fX(Math.max(...r));
    return "同形状 kernel 本页 "+(lo===hi?lo:lo+"–"+hi)+"（FP8 峰值是 BF16 的 2 倍，再加反量化），实测只有 1.4–1.8×：W4A8 kernel 离 FP8 峰值更远。Qwen3.6 形状一层 MoE（激活单独量化）T = 64 / 1024："+fX(e2[0])+" / "+fX(e2[1])+"，方向和端到端慢 3–8% 一致"; }
  if(key==="fuse"){ const h=HW.GB300, {m}=ctx(), topo=TOPO.nvl64, base=Object.assign({},S.inf,{phase:"decode",N:4,dpA:1,moePar:"tp",wDt:"nvfp4",eAct:"",dDt:"bf16",specK:0});
    const r=[1,16,128].map(B=>{ const a=Infer.decode(m,h,topo,Object.assign({},base,{B,qSep:true}),g), b=Infer.decode(m,h,topo,Object.assign({},base,{B,qSep:false}),g); return (a.stepT/b.stepT-1)*100; });
    return "当前模型、GB300 TP4、NVFP4 expert：并发 1 / 16 / 128 融合后快 "+r.map(x=>x.toFixed(1)+"%").join(" / "); }
  if(key==="mem"){ const M=2048,K=2048,N=8192, d=Q.trainGemm(HW.B200,g,"mxfp8",M,N,K,{nMicro:1}), h=Q.trainGemm(HW.B200,g,"mxfp8hp",M,N,K,{nMicro:1}),
      dn=Q.trainGemm(HW.B200,g,"nvfp4",M,N,K,{nMicro:1}), hn=Q.trainGemm(HW.B200,g,"nvfp4hp",M,N,K,{nMicro:1});
    return "按副本数算：MXFP8 "+fB(d.extra)+" → "+fB(h.extra)+"（"+((h.extra/d.extra-1)*100).toFixed(0)+"%，方向一致）；NVFP4 默认 "+fB(dn.extra)+" → "+fB(hn.extra)+"：方向相反，TE 默认路径的峰值来自 RHT / 2D 量化的临时 buffer，这里没建模"; }
  if(key==="trainRecipe"){ const {m}=ctx(), e=m.moe, De=e.Dlat||m.D, F=e.F||m.dense.F, M=8192, t=k=>Q.trainGemm(HW.B200,g,k,M,2*F,De,{nMicro:4,tokFused:false}).t, b=t("bf16");
    return "当前模型 "+(e.E?"expert":"dense FFN")+" gate_up、M=8192：MXFP8 "+fX(b/t("mxfp8"))+"，NVFP4 + BF16 反向（per-token 未融合）"+fX(b/t("nvfp4hp"))+"，融合后 "+fX(b/Q.trainGemm(HW.B200,g,"nvfp4hp",M,2*F,De,{nMicro:4,tokFused:true}).t)+"（相对 BF16 的速度）"; }
  if(key==="moeTrain"){ const D=5120, F=8192, M=16384, t=k=>Q.trainGemm(HW.B200,g,k,M,2*F,D,{nMicro:1}).t+Q.trainGemm(HW.B200,g,k,M,D,F,{nMicro:1}).t;
    return "Llama4 Scout expert（D 5120、F 8192，每 expert 16K 行）只算 grouped GEMM：MXFP8 "+fX(t("bf16")/t("mxfp8"))+"；实测 1.40× 里还有 attention、a2a 等没加速的部分"; }
  return "";
}

/* 各节共用：填两个方案选择器、硬件说明 */
function qHead(){
  const q=S.q, hw=HW[q.hw], Q=Quant, SC=Q.SCHEMES;
  fillOpts($("q-scheme"), Q.ORDER.map(k=>[k,SC[k].short+"："+SC[k].name]), q.scheme);
  if(!Q.RORDER.includes(q.recipe)) q.recipe="mxfp8";
  fillOpts($("q-recipe"), Q.RORDER.map(k=>[k,Q.RECIPES[k].name]), q.recipe);
  $("q-hw-note").textContent = hw.name+"：BF16 "+hw.bf16/1000+" / FP8 "+hw.fp8/1000+" / FP4 "+hw.fp4/1000+" PFLOPS（dense），HBM "+hw.hbm/1000+" TB/s，"+hw.mem+" GB。"+
    (/^GB/.test(q.hw)?"放回整个模型时按 NVL72 机柜算。":"放回整个模型时按 HGX 算。")+"推理页 / 后训练页的配置照搬，只换精度。";
  return { q, hw, g:S.glob, Q, SC };
}
const qMs = (hw,g,N,K,o) => { const Ms={}; Quant.ORDER.forEach(k=>Ms[k]=Quant.crossover(hw,g,k,N,K,o)); return Ms; };
const qLegend = cur => Quant.ORDER.map(k=>'<span style="color:'+QC[k]+'"><i'+(k===cur||k==="bf16"?'':' class="d"')+'></i>'+esc(Quant.SCHEMES[k].short)+'</span>').join("");

/* 推理：一个 GEMM、一层 MoE、放回整个 decode 一步 */
function renderQuantInfer(){
  const {m,d}=ctx(), {q,hw,g,Q,SC}=qHead(), o=qOpt();
  const N=q.N, K=q.K, M=Math.max(1,q.M), cur=q.scheme, G={};
  Q.ORDER.forEach(k=>G[k]=Q.gemm(hw,g,k,M,N,K,o));
  const Ms=qMs(hw,g,N,K,o), x=G[cur], b=G.bf16;
  $("q-lede").innerHTML="Y["+fInt(M)+","+fInt(N)+"] = X["+fInt(M)+","+fInt(K)+"]·W["+fInt(N)+","+fInt(K)+"]ᵀ，"+esc(SC[cur].name)+"，"+esc(hw.name)+"。"+
    (x.bound==="memory"?"这个 M 下字节决定时间，权重占 "+pct(x.wB,x.bytes)+"。":x.bound==="compute"?"这个 M 下算力决定时间，MMA 跑在 "+SC[cur].pk.toUpperCase()+" 峰值上。":"这个 GEMM 太小，时间就是一个 kernel 的最短时间。");
  kpis($("q-kpis"),[
    ["GEMM 时间",fT(x.t),x.tQ>0?"其中激活量化 "+fT(x.tQ):x.tDeq>0?"其中反量化 "+fT(x.tDeq):""],
    ["相对 BF16",fX(b.t/x.t),"BF16 "+fT(b.t),b.t<x.t],
    ["瓶颈",x.bound,"I = "+fI(x.I)+" FLOP/B"],
    ["拐点 M*",isFinite(Ms[cur])?fInt(Ms[cur]):"∞","BF16 "+fInt(Ms.bf16)],
    ["权重字节",fB(x.wB),"其中 scale "+fB(x.wScale)],
    ["激活字节",fB(x.aB+x.qB),x.qB?"含量化 kernel 读写 "+fB(x.qB):""]]);
  // 扫 M
  const xs=logspace(1,65536,120), series=[], vl=[{x:M,color:css("ink"),label:"当前 M = "+fInt(M)}];
  for(const k of Q.ORDER){ series.push({pts:xs.map(v=>[v, Q.gemm(hw,g,"bf16",v,N,K,o).t/Q.gemm(hw,g,k,v,N,K,o).t]),color:QC[k],w:k===cur?3:1.5,dash:k===cur||k==="bf16"?null:"6 4",label:k===cur?SC[k].short:null}); }
  if(isFinite(Ms[cur])) vl.push({x:Ms[cur],color:QC[cur],label:"M* "+fInt(Ms[cur]),dy:14});
  plot($("q-sweep"),{x0:1,x1:65536,y0:0.25,y1:hw.fp4/hw.bf16*1.6,logx:true,logy:true,fx:fxPow,fy:v=>+v.toPrecision(2)+"×",xl:"M（行数）",yl:"相对 BF16 的速度",series,vlines:vl});
  const pks=[{v:peakOf(hw,"bf16"),n:"BF16"},{v:peakOf(hw,"fp8"),n:"FP8"},{v:peakOf(hw,"fp4"),n:"FP4"}];
  plot($("q-roof"),{x0:1,x1:30000,y0:1e11,y1:pks[2].v*3,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:"算术强度 I（FLOP / HBM 字节）",yl:"实际达到的 FLOP/s",
    series:roofSeries(hw,g,pks,1,30000),points:Q.ORDER.map(k=>({x:G[k].I,y:G[k].flReal/G[k].t,color:QC[k],r:k===cur?6:4,label:SC[k].short}))});
  $("q-sweep-legend").innerHTML=qLegend(cur)+'<span>右图三条屋顶：BF16 / FP8 / FP4 峰值 × η<sub>c</sub></span>';
  table($("q-gemm"),["方案","W / A 位","权重","其中 scale","激活","计算","访存","量化 kernel","反量化","合计","瓶颈","相对 BF16","M*"],
    Q.ORDER.map(k=>{ const r=G[k]; return {cls:k===cur?"hl":"",cells:[sw(QC[k])+esc(SC[k].short),bits(SC[k].w)+" / "+bits(SC[k].a),fB(r.wB),fB(r.wScale),fB(r.aB),fT(r.tc),fT(r.tm),r.tQ?fT(r.tQ):"—",r.tDeq?fT(r.tDeq):"—",fT(r.t),pill(r.bound),fX(b.t/r.t),isFinite(Ms[k])?fInt(Ms[k]):"∞"]}; }),[2,3,4,5,6,7,8,9,11,12]);
  // MoE
  const e=m.moe, De=e.Dlat||m.D, T=Math.max(1,q.T), ep=Math.max(1,q.ep), oM=qOpt({align:q.align,floor:true}), L={};
  const hasMoE=e.E>0; let lam=0, Lc={a:"—",rows:0,rowsPad:0,bound:"memory"};
  ["q-moe","q-moe-legend","q-moe-tbl","q-align"].forEach(id=>{ const el=$(id); (el.closest(".fig,.tbl")||el).hidden=!hasMoE; });
  if(!hasMoE) $("q-moe-lede").innerHTML=esc(m.short)+" 是 dense 模型，没有 routed expert：每个 FFN GEMM 的 M 就是整批 token 数，不会被 E/k 摊薄，也没有对齐补齐。上面单个 GEMM 一节选 gate_up / down 看到的就是它的 dense FFN。";
  else {
  Q.ORDER.forEach(k=>L[k]=Q.moeLayer(hw,g,m,k,T,ep,oM));
  lam=T*e.k/e.E; Lc=L[cur];
  $("q-moe-lede").innerHTML=esc(m.short)+"：E = "+e.E+"，top-"+e.k+"，expert 是 "+De+" → "+e.F+"。这一层看到 T = "+fInt(T)+" 个 token 时，平均每个 expert 只有 λ = T·k/E = "+V(fI(lam))+" 行，"+
    (lam<1?"大部分 expert 一行都没有，碰到的 expert 各读一遍权重，几乎全是 memory-bound：MoE 等于把每个 GEMM 的 M 除以 E/k。":lam<64?"每个 expert 的 GEMM 仍在拐点左边，量化权重最划算。":"每个 expert 已经接近或越过拐点，激活也要量化才继续有收益。")+
    "下图按均匀路由（每个 expert 的行数服从 Poisson(λ)）算整层 routed expert 的时间，含对齐补齐；真实路由更偏，冷 expert 更多、热 expert 更满。";
  const Ts=logspace(1,262144,90), ms=[];
  for(const k of Q.ORDER) ms.push({pts:Ts.map(t=>[t, Q.moeLayer(hw,g,m,"bf16",t,ep,oM).t/Q.moeLayer(hw,g,m,k,t,ep,oM).t]),color:QC[k],w:k===cur?3:1.5,dash:k===cur||k==="bf16"?null:"6 4",label:k===cur?SC[k].short:null});
  const tAt = l => l*e.E/e.k;
  plot($("q-moe"),{x0:1,x1:262144,y0:0.25,y1:hw.fp4/hw.bf16*1.6,logx:true,logy:true,fx:fxPow,fy:v=>+v.toPrecision(2)+"×",xl:"这一层看到的 token T（整个 EP 组）",yl:"整层 routed expert 相对 BF16",series:ms,
    vlines:[{x:T,color:css("ink"),label:"当前 T"},{x:tAt(1),color:css("muted"),label:"λ = 1",dy:14},{x:tAt(Ms.bf16),color:css("muted"),label:"λ = BF16 的 M*",dy:28}]});
  $("q-moe-legend").innerHTML=qLegend(cur);
  table($("q-moe-tbl"),["方案（EP "+ep+"，每 rank "+fI(e.E/ep)+" 个 expert）","每 expert 行数 λ","tile / 对齐","补齐后行数","补齐浪费","碰到的 expert","权重读取","时间","瓶颈","相对 BF16"],
    Q.ORDER.map(k=>{ const r=L[k]; return {cls:k===cur?"hl":"",cells:[sw(QC[k])+esc(SC[k].short),fI(r.lam),r.a,fN(r.rowsPad),pct(r.rowsPad-r.rows,r.rowsPad),fI(r.touched)+" / "+fI(r.El),fB(r.wB),fT(r.t),pill(r.bound),fX(L.bf16.t/r.t)]}; }),[1,2,3,4,5,6,7,9]);
  const AL=[[1,"推理 masked 布局（DeepGEMM masked_m）：不补行，但 MMA 仍按 tile 算"],[8,"FlashInfer TRTLLM MoE 的最小 tile"],[16,"FP8 训练：Megatron --moe-router-padding-for-quantization"],[32,"MXFP8 / NVFP4 训练：scale 组按 32 行；torchtitan pad_multiple=32"],[64,"FlashInfer TRTLLM MoE 的最大 tile（近似）"],[128,"DeepGEMM contiguous 布局；torchtitan CuTeDSL 路径"]];
  table($("q-align"),["每 expert 行数补到","补齐后期望行数（λ = "+fI(lam)+"）","浪费的行","在哪里"],
    AL.map(([a,w])=>{ const p=Q.poisson(lam,a,q.BM); return {cls:String(a)===String(Lc.a)?"hl":"",cells:[a,fN(e.E*p.pad),pct(Math.max(0,p.pad-lam),p.pad),esc(w)]}; }),[0,1,2]);
  }
  // 放回整个 decode 一步
  const IE={}; Q.ORDER.forEach(k=>IE[k]=qInferE2E(hw,k));
  $("q-e2e-inf-lede").innerHTML="照搬推理页的 decode 配置（"+esc(ENGN[S.inf.engine]||S.inf.engine)+"，N = "+S.inf.N+"，B = "+S.inf.B+"，上下文 "+fInt(S.inf.n)+"），只把精度换掉、GPU 换成 "+esc(hw.name)+"。"+
    "只量化 expert 的方案（MXFP4 / NVFP4）dense 部分保持 BF16"+(m.moe.E?"。":"；当前是 dense 模型，没有 expert，所以把方案作用到全部线性层（lm_head 除外）。");
  const ib=IE.bf16;
  table($("q-e2e-inf"),["推理：方案","expert / 其余权重","每卡权重","显存允许的并发","decode 一步","每卡 tok/s","相对 BF16"],
    Q.ORDER.map(k=>{ const r=IE[k], c=QE2E[k], fit=r.mem.avail>0, over=fit&&S.inf.B>r.mem.Bmax;
      return {cls:k===cur?"hl":"",cells:[sw(QC[k])+esc(SC[k].short),DTN[c.wDt]+(c.eAct?"（激活 "+DTN[c.eAct]+"）":"")+" / "+DTN[c.dDt],fB(r.mem.weights),fit?fInt(r.mem.Bmax)+(over?' <span class="tag">B 超出</span>':""):'<span class="tag">放不下</span>',fT(r.stepT),fN(r.tokPerGpu),fX(ib.stepT/r.stepT)]}; }),[2,3,4,5,6]);
  // 解析
  const A=[], ec=d.pExpertsAll/d.total;
  const big16=Q.gemm(hw,g,"bf16",16384,N,K,o), big4=Q.gemm(hw,g,"w4a16",16384,N,K,o);
  A.push("拐点 M*：BF16 "+V(fInt(Ms.bf16))+"、W4A16 "+V(fInt(Ms.w4a16))+"、W8A8 "+V(fInt(Ms.mxfp8))+"、W4A4 "+V(fInt(Ms.nvfp4))+"。M 在拐点左边时，只要权重变窄就快；M = 16K 时 W4A16 是 BF16 的 "+V(fX(big16.t/big4.t))+"，因为 MMA 还是 BF16，每 "+q.BM+" 行还要反量化一遍权重（反量化吞吐是可调的经验值）。");
  if(!hasMoE) A.push(esc(m.short)+" 是 dense 模型：decode 每步要读全部 "+V(fN(d.total))+" 参数，量化权重直接按比例省带宽；但同样的 batch 下每个 GEMM 的 M 就是整批 token，比 MoE 的每 expert 行数大 E/k 倍，更早越过拐点，所以大 batch 时只有激活也降精度（W8A8 / W4A4）才继续有收益。");
  else A.push(esc(m.short)+" 的 routed expert 占总参数 "+V(pct(d.pExpertsAll,d.total))+"："+(ec>0.8?"只量化 expert 就拿到大部分显存和带宽收益，这也是 NVFP4 rollout 只量化 expert 的原因（<a href=\"#rl/lowprec\">RL · 低精度 rollout</a>）。":"非 expert 部分也不小，只量化 expert 的收益有限。")+"但按每 token 实际经过的参数算，routed expert 只占 "+V(pct(d.nMoe*m.moe.k*d.pExpert,d.active))+"，所以省的主要是显存和权重带宽，不是算力；留在 BF16 的 router、shared expert、注意力投影和 lm_head 恰好也是量化后最敏感的部分。当前 T 下每个 expert 只有 "+V(fI(lam))+" 行，补齐到 "+Lc.a+" 行浪费 "+V(pct(Lc.rowsPad-Lc.rows,Lc.rowsPad))+" 的算力"+(Lc.bound==="memory"?"，但这一层是 memory-bound，浪费的算力不影响时间。":"，这一层已经是 compute-bound，补齐直接变成时间。"));
  const qs=Q.gemm(hw,g,cur,M,N,K,qOpt({fuse:"separate"})), qf=Q.gemm(hw,g,cur,M,N,K,qOpt({fuse:"fused"}));
  if(SC[cur].a!=="bf16") A.push("激活单独量化时，按纯 roofline 这个 GEMM 多 "+V(fT(qs.tQ))+"（"+pct(qs.tQ,qf.t)+"），字节是 M·K·(2 + 量化后字节)，随 M 线性涨。M 小时真正的代价是多一个 kernel 的最短时间和 launch：SGLang 把 RMSNorm 和 NVFP4 量化融合后，并发 1 快 6%、并发 128 只快 1%（#44554）。hidden 小的模型上这笔字节又摊不开：vLLM 在 hidden 2048 的 Qwen3.6-35B-A3B 上 W4A8 端到端反而比 W4A16 慢 3–8%（#59815）。");
  const bestI=Q.ORDER.filter(k=>k!=="bf16"&&IE[k].mem.avail>0).sort((a,b)=>IE[a].stepT-IE[b].stepT)[0];
  if(bestI) A.push("放回整个 decode 一步，最快的是 "+esc(SC[bestI].short)+" "+V(fX(ib.stepT/IE[bestI].stepT))+"：注意力读 KV、通信、launch 这些不随权重精度变，单个 GEMM 的加速在整步里被摊薄。量化还多换来显存：同样的卡，BF16 并发上限 "+V(fInt(ib.mem.Bmax))+"，"+esc(SC[bestI].short)+" "+V(fInt(IE[bestI].mem.Bmax))+"。");
  A.push("B300 和 B200 只差 FP4 峰值（13.5 对 9 PF）和显存（288 对 180 GB）：拐点以左两者一样快，所以小并发 decode 换 B300 不会更快，只能放更多请求；GB200 / GB300 是同一颗芯片，BF16 / FP8 峰值高约 11%，主要差别在 NVL72 的 a2a 和权重同步。");
  A.push("这一节只看精度对推理本身的影响。RL 的 rollout 还要和训练侧量化逐位一致、每步同步权重，见 <a href=\"#rl/lowprec\">RL · 低精度 rollout</a>；低精度 kernel 的 batch invariance 见 <a href=\"#rl/determinism\">RL · 确定性 rollout</a>。");
  $("q-analysis-inf").innerHTML=ul(A);
}

/* 后训练：三遍 GEMM、放回整个 step */
function renderQuantTrain(){
  const {q,hw,g,Q}=qHead(), N=q.N, K=q.K;
  const Mt=Math.max(16,q.Mt), TR={}, to={nMicro:q.nMicro,tokFused:q.tokFused};
  Q.RORDER.forEach(k=>TR[k]=Q.trainGemm(hw,g,k,Mt,N,K,to));
  const ideal=k=>{ const R=Q.RECIPES[k], rf=peakOf(hw,R.f)/peakOf(hw,"bf16"), rb=peakOf(hw,R.b)/peakOf(hw,"bf16"); return (1/rf+2/rb)/3; };
  table($("q-train"),["配方","fprop","dgrad","wgrad","量化相关 kernel","合计","相对 BF16","只看峰值","反向要留的（X + W 缓存）","谁在用"],
    Q.RORDER.map(k=>{ const r=TR[k]; return {cls:k===q.recipe?"hl":"",cells:[esc(r.R.name),fT(r.fprop),fT(r.dgrad),fT(r.wgrad),r.tq?fT(r.tq):"—",fT(r.t),fX(TR.bf16.t/r.t),fX(1/ideal(k)),fB(r.extra),esc(r.R.who)]}; }),[1,2,3,4,5,6,7,8]);
  const tr=TR[q.recipe];
  stackBar($("q-tbar"),$("q-tbarlist"),[{label:"fprop（"+DTN[tr.R.f]+"）",v:tr.fprop,color:css("c-attn")},{label:"dgrad（"+DTN[tr.R.b]+"）",v:tr.dgrad,color:css("c-moe")},{label:"wgrad（"+DTN[tr.R.b]+"）",v:tr.wgrad,color:css("c-lin")},
    ...tr.q.map((z,i)=>({label:z[0],v:z[1],color:[css("c-cp"),css("c-a2a"),css("c-ar"),css("c-dp"),css("muted")][i%5]}))],tr.t,fT);
  // 放回整个 step
  const TE={}; QTRN.forEach(k=>TE[k]=qTrainE2E(hw,k));
  $("q-e2e-trn-lede").innerHTML="照搬后训练页的配置（"+(S.trn.fw==="megatron"?"Megatron-Core":"torchtitan")+"，TP "+S.trn.tp+" / CP "+S.trn.cp+" / EP "+S.trn.ep+" / PP "+S.trn.pp+"，每卡 "+fInt(S.trn.tok)+" token），只把精度换掉、GPU 换成 "+esc(hw.name)+"。精度作用于所有 GEMM（包括 lm_head），MoE 的对齐补齐没有算进去。";
  const tb=TE.bf16;
  table($("q-e2e-trn"),["训练：GEMM 精度","step 时间","每卡 tok/s","相对 BF16","低精度权重缓存","显存"],
    QTRN.map(k=>{ const r=TE[k]; return {cls:k===q.recipe?"hl":"",cells:[esc(Train.PREC[k].name),fT(r.stepT),fN(r.tokPerGpu),fX(tb.stepT/r.stepT),fB(r.mem.qW),fB(r.mem.used)+(r.mem.used>r.mem.total?' <span class="tag">超出</span>':"")]}; }),[1,2,3,4,5]);
  // 解析
  const A=[], best=Q.RORDER.filter(k=>k!=="bf16").sort((a,b)=>TR[a].t-TR[b].t)[0], mx=TR.mxfp8, tt=TR.mxfp8tt, wq=r=>(r.q.find(z=>/^W 量化/.test(z[0]))||[0,0])[1];
  A.push("单个 GEMM（M = "+fInt(Mt)+"）：最快的是 "+esc(Q.RECIPES[best].name)+" "+V(fX(TR.bf16.t/TR[best].t))+"，只看峰值是 "+V(fX(1/ideal(best)))+"：差的是量化 kernel 和权重的转置副本。"+
    "MXFP8 两种权重块只差在权重上：TE 默认的 1D 块要行、列两份（缓存 "+fB(mx.wCache)+"），torchtitan 的 32×32 方块一份（"+fB(tt.wCache)+"）；权重量化每个 step 做一次、按 "+q.nMicro+" 个 micro-batch 摊开，时间上只差 "+fT(wq(mx)-wq(tt))+"。");
  const bestT=QTRN.filter(k=>k!=="bf16").sort((a,b)=>TE[a].stepT-TE[b].stepT)[0];
  A.push("放回整个 step，最快的 "+esc(Train.PREC[bestT].name)+" 只快 "+V(fX(tb.stepT/TE[bestT].stepT))+"：注意力核心、线性注意力、通信、优化器和逐元素算子都不随 GEMM 精度变（Amdahl）。");
  A.push("这一节只看默认配方对训练速度的影响。RL 里训练要和低精度 rollout 对齐，常用前向低精度、反向 BF16（<code>high_precision</code> / <code>dequantized</code>）和 QAT，这些放在 <a href=\"#rl/qat\">RL · 低精度训练与 QAT</a>；整个 RL 一步的账见 <a href=\"#rl/lowprec\">RL · 低精度 rollout</a>。");
  $("q-analysis-trn").innerHTML=ul(A);
}

/* 附录 · 量化基础：格式表 */
function renderFormats(){
  const {m,d}=ctx();
  const binE2=Math.log2(6/0.5), binE4=Math.log2(448/Math.pow(2,-9)), wAll=m.moe.E?d.pExpertsAll:d.total-d.pEmb-d.pHead;
  table($("q-formats"),["格式","元素","块（沿 K）","块 scale","第二级","每元素位","元素动态范围","scale 的代价",m.moe.E?"当前模型全部 routed expert":"当前模型全部线性层（不含词表）"],[
    ["按张量 / 通道 / token FP8","E4M3","无","FP32","—","8","≈ "+binE4.toFixed(1)+" 档","outlier 拉低整行精度；scale 可提到 epilogue",fB(wAll*1)],
    ["block FP8","E4M3","权重 128×128，激活 1×128","FP32（Blackwell 上取整成 UE8M0）","—","权重 "+bits("fp8")+"，激活 "+bits("fp8a"),"≈ "+binE4.toFixed(1)+" 档","Blackwell 上要 requant 成 2 的幂",fB(wAll*Common.dtB("fp8"))],
    ["MXFP8","E4M3","1×32（torchtitan 权重 32×32）","E8M0","—",bits("mxfp8"),"≈ "+binE4.toFixed(1)+" 档","向上取整丢约半档，相对 17.8 档可忽略",fB(wAll*Common.dtB("mxfp8"))],
    ["MXFP4","E2M1","1×32","E8M0","—",bits("mxfp4"),"≈ "+binE2.toFixed(1)+" 档","丢的半档相对 3.6 档很可观",fB(wAll*Common.dtB("mxfp4"))],
    ["NVFP4","E2M1","1×16（TE 预训练权重 16×16）","E4M3","FP32：amax / (448 × 6)，按张量或按 token",bits("nvfp4"),"≈ "+binE2.toFixed(1)+" 档","块 amax 能对齐到 6；4over6 再在 6 和 4 之间选 MSE 小的",fB(wAll*Common.dtB("nvfp4"))]],[5,8]);
}

/* RL：低精度 rollout 和训练组合成一步（Amdahl） */
function qRL(){
  const {q,hw,Q,SC}=qHead(), f=Math.min(1,Math.max(0,q.roll)), sp=(sr,st)=>1/(f/sr+(1-f)/st);
  const IE={}, TE={}; ["bf16","mxfp8","nvfp4","w4a16"].forEach(k=>IE[k]=qInferE2E(hw,k)); ["bf16","mxfp8","mxfp8hp","nvfp4","nvfp4hp"].forEach(k=>TE[k]=qTrainE2E(hw,k));
  const RLC=[["BF16 训练 + BF16 rollout","bf16","bf16","是"],["MXFP8 端到端","mxfp8","mxfp8","是"],["MXFP8 rollout + MXFP8 前向 / BF16 反向","mxfp8","mxfp8hp","是"],
    ["NVFP4 expert rollout + NVFP4 前向 / BF16 反向（Miles 脚本；各种反向一个 GEMM 的开销见低精度训练与 QAT 一节）","nvfp4","nvfp4hp","是：1D 1×16、按 token"],["NVFP4 expert rollout + TE 默认 NVFP4 训练","nvfp4","nvfp4","否：训练权重 16×16，rollout 1×16"],
    ["W4A16 rollout + BF16 训练","w4a16","bf16","否：训练是 BF16，只能靠 QAT 假量化补（slime INT4 配方）"]];
  table($("q-rl"),["RL 组合","rollout 加速","训练加速","一步加速（f = "+f+"）","两侧量化一致"],
    RLC.map(([n,a,t,c])=>{ const sr=IE.bf16.stepT/IE[a].stepT, st=TE.bf16.stepT/TE[t].stepT; return [esc(n),fX(sr),fX(st),fX(sp(sr,st)),esc(c)]; }),[1,2,3]);
  return { hw, f, IE, TE, SC };
}
