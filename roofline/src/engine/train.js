/* 训练核心：Megatron-Core 与 torchtitan 的逐层 fwd / dgrad / wgrad / 重算，以及各自的并行通信。
   与推理核心不共享算子表：这里没有 KV cache、状态缓存、采样，有的是反向、重算、整段 CE、优化器与梯度同步。 */
const Train = (function(){
const { dtB, peakOf, derive, qoW, swaKeys, link, agT, arT, a2aT, roof } = Common;

/* FLA chunk kernel（C=64）每个 value head、每个 token 的 FLOPs：前向 / 反向（反向重算 w,u 与整条 h 递推） */
function linFlops(kind, K, V){ const C=64;
  return kind==="KDA" ? { f:6*K*V+3.5*C*K+4*C*V+0.5*C*C, b:16*K*V+9*C*K+10*C*V+4*C*C }
                      : { f:6*K*V+7.25*C*K+4*C*V+0.5*C*C, b:16*K*V+18*C*K+10*C*V+4*C*C }; }

/* GEMM 精度配方：f 是前向，b 是 dgrad / wgrad；w2d 表示权重用方块，一份量化同时给两个方向用 */
const PREC = {
  bf16:   { f:"bf16",  b:"bf16",  name:"BF16" },
  fp8:    { f:"fp8",   b:"fp8",   w2d:true, name:"block FP8（权重 128×128，激活 1×128）" },
  mxfp8:  { f:"mxfp8", b:"mxfp8", name:"MXFP8（TE 默认，1D 行列各一份）" },
  mxfp8hp:{ f:"mxfp8", b:"bf16",  name:"MXFP8 前向 + BF16 反向" },
  nvfp4:  { f:"nvfp4", b:"nvfp4", w2d:true, name:"NVFP4（TE 预训练默认，权重 16×16）" },
  nvfp4hp:{ f:"nvfp4", b:"bf16",  tok:true, name:"NVFP4 前向 + BF16 反向（Miles 脚本默认）" } };

function run(m, hw, topo, c, g){
  const d=derive(m), f=m.full, l=m.lin, e=m.moe, D=m.D, N=c.N, mg=c.fw==="megatron";
  const tp=c.tp, cp=c.cp, pp=c.pp, vpp=Math.max(1,c.vpp||1), ep=c.ep, etp=mg?Math.max(1,c.etp||1):1;
  const dp = N/(tp*cp*pp);
  const edp = mg ? N/(etp*ep*pp) : dp*cp*tp/ep;
  const warn=[];
  if(!Number.isInteger(dp)) warn.push("TP×CP×PP 必须整除 "+N);
  if(mg && !Number.isInteger(edp)) warn.push("ETP×EP×PP 必须整除 "+N);
  if(d.nMoe>0 && !mg && ep<tp) warn.push("torchtitan 要求 EP ≥ TP");
  if(d.nMoe>0 && e.E%ep) warn.push("EP 最好整除 expert 数 "+e.E);
  const tok=c.tok, T=tok*cp, Tc=tok, Tl=tok/tp;    // T：TP×CP 组一个 micro-batch 的 token；Tc：本 CP rank；Tl：SP 下本 rank 驻留
  const PR=PREC[c.prec]||PREC.bf16, fq=PR.f!=="bf16", bq=PR.b!=="bf16";
  const S=c.S, wbF=dtB(PR.f), wbB=dtB(PR.b), pkF=peakOf(hw,PR.f), pkB=peakOf(hw,PR.b), pkA=peakOf(hw,"bf16");
  const pkAt = c.etaA>0 ? pkA*c.etaA/g.etaC : pkA;          // attention 核心单独给效率（0 = 和 GEMM 一样用 η_c）
  if(fq && PR.f!=="fp8" && hw.sm<100) warn.push(PR.name+" 需要 Blackwell 的块缩放 MMA；Hopper 上这里只按 FP8 峰值粗算");
  // 低精度时每个 GEMM 输入多一次量化：读 bf16，写行方向一份（反向也低精度时再写列方向一份，wgrad 用）
  const qF = fq ? 2 + wbF + (bq?wbB:0) : 0, qB = bq ? 2 + 2*wbB : 0;
  const ops=[], comm=[];
  let tokB=0;                                       // NVFP4 按 token FP32 scale 没融合时，每个 GEMM 前后各多一次逐元素缩放
  // 每个算子三遍：fwd、dgrad（反向对输入）、wgrad（反向对权重）；非 GEMM 的反向全记在 dgrad
  const op=(key,name,group,layers,fwd,dg,wg,extra)=>{ if(layers<=0) return; if(fwd&&fwd.tb) tokB+=fwd.tb*layers;
    const r=(x,pk)=>x? roof(x.fl, x.by, pk||x.pk, hw, g, x.k||1) : {t:0,tc:0,tm:0,bound:"—",I:0};
    const rf=r(fwd), rd=r(dg), rw=r(wg);
    ops.push(Object.assign({key,name,group,layers,
      fwd:{fl:(fwd?fwd.fl:0)*layers, by:(fwd?fwd.by:0)*layers, t:rf.t*layers, bound:rf.bound, I:rf.I},
      dgrad:{fl:(dg?dg.fl:0)*layers, by:(dg?dg.by:0)*layers, t:rd.t*layers, bound:rd.bound, I:rd.I},
      wgrad:{fl:(wg?wg.fl:0)*layers, by:(wg?wg.by:0)*layers, t:rw.t*layers, bound:rw.bound, I:rw.I}},extra||{})); };
  // GEMM：M 行、K→N，权重按 shard 切；三遍 FLOPs 相同，字节不同
  const gemm=(M,K,Nn,wShard)=>{ const W=K*Nn/wShard;
    return [ {fl:2*M*K*Nn/wShard, by:W*wbF + M*(K+Nn/wShard)*2 + qF*M*K, pk:pkF, tb: PR.tok&&!c.tokFused ? 4*M*K + 4*M*Nn/wShard : 0},
             {fl:2*M*K*Nn/wShard, by:W*wbB + M*(K+Nn/wShard)*2 + qB*M*Nn/wShard, pk:pkB},
             {fl:2*M*K*Nn/wShard, by:W*4*2 + M*(K+Nn/wShard)*2, pk:pkB} ]; };      // wgrad 累加进 fp32 main_grad
  const elem=(flPerTok,byPerTok,M,bwdMult)=>[ {fl:flPerTok*M,by:byPerTok*M,pk:pkA}, {fl:flPerTok*M*(bwdMult||2),by:byPerTok*M*(bwdMult||2),pk:pkA}, null ];
  const cm=(key,name,cls,layers,passes,t1,bytes1,extra)=>{ if(layers<=0||t1<=0) return; comm.push(Object.assign({key,name,cls,layers,passes,t:t1*layers,bytes:bytes1*layers},extra||{})); };

  const lkTP=link(hw,topo,tp,1), lkCP=link(hw,topo,cp,tp), lkEP=link(hw,topo,ep, mg?etp:1);
  const lkPP=link(hw,topo,2,tp*cp*dp), lkDP=link(hw,topo,dp*cp,tp);
  const lkEDP=link(hw,topo,edp, mg?etp*ep:ep);
  // 重算策略 → 每个算子是否在反向里再跑一次 fwd
  const rcFull = c.recompute==="full";
  const sel = new Set(c.rcModules||[]);
  const ttDef = !mg && c.recompute==="default";
  const rc = key => rcFull ? true : mg ? (
      (sel.has("core_attn") && key==="acore") || (sel.has("layernorm") && key==="norm") || (sel.has("moe_act") && key==="act") ||
      (sel.has("mla_up_proj") && (key==="aqb"||key==="akvb")) || (sel.has("moe") && ["router","permute","moe1","act","moe2","unperm","latd","latu"].includes(key)) ||
      (sel.has("shared_experts") && key==="shared") || (sel.has("gdn") && ["lproj","lconv","lcore","lnorm","lout"].includes(key)) )
    : ttDef ? (key==="moe1"||key==="act") : false;
  const rcA2A = mg ? (rcFull || sel.has("moe")) : (rcFull && c.dispatcher!=="distmoe");
  const tpBlocks = []; // 走 TP+SP 的块：每块 fwd AG+RS，bwd 3 AG + 1 RS

  /* ---- 全注意力 ---- */
  let coreFullOp=null, coreSwaOp=null;
  if(f.n>0){
    op("norm","pre-attn RMSNorm","misc",f.n,...elem(8*D/tp,3*D*2/tp,Tc,2));
    if(f.kind==="gqa"){
      const qo=qoW(f), kvo=2*f.Hkv*f.dh, nSwa=f.swa||0;
      op("aqkv","QKV"+(f.gate?"（含门）":"")+" GEMM","attn",f.n,...gemm(Tc,D,qo+kvo,tp));
      op("arope","q/k norm + RoPE","attn",f.n,...elem(10*(f.Hq+2*f.Hkv)*f.dh/tp,(f.Hq+2*f.Hkv)*f.dh*2*2/tp,Tc,2));
      // 每 token 平均看 S/2 个 key（滑窗层看 swaKeys 个）：QKᵀ + PV
      const core=(layers,keys,tag)=>{ const fl=4*(f.Hq/tp)*f.dh*keys*Tc;
        op("acore","causal attention（"+(mg?"TE fused / cuDNN":"varlen：FA3 / FA4 / cuDNN")+"）"+tag,"attn",layers,{fl,by:Tc*(2*f.Hq+2*f.Hkv)*f.dh*2/tp,pk:pkAt},{fl:2.5*fl,by:Tc*(4*f.Hq+4*f.Hkv)*f.dh*2/tp,pk:pkAt},null); return ops[ops.length-1]; };
      coreFullOp = core(f.n-nSwa, S/2, nSwa?"：全局层":"");
      if(nSwa) coreSwaOp = core(nSwa, swaKeys(S,f.W), "：滑窗 "+f.W);
      if(f.gate) op("agate",f.gateHead?"每头 sigmoid 门":"sigmoid 输出门","attn",f.n,...elem(2*f.Hq*f.dh/tp,3*f.Hq*f.dh*2/tp,Tc,2));
      op("aout","o_proj GEMM","attn",f.n,...gemm(Tc,f.Hq*f.dh,D,tp));
    } else {
      const H=f.H, dq=f.nope+f.rope, lat=f.kvLora+f.rope;
      op("aqa","q / kv 下投影（"+(mg?"duplicated，SP 本地 token":"wq_a / wkv_a 复制")+"）","attn",f.n,...gemm(mg?Tl:Tc,D,f.qLora+lat,1));
      op("aqb","q_b 上投影","attn",f.n,...gemm(Tc,f.qLora,H*dq,tp));
      if(f.kind==="dsa"){
        // Megatron DSA = absorbed MLA：MQA 一个 latent 头；indexer 在每个 TP rank 上对整个 TP 组序列重复算
        op("auk","W_uk absorb","attn",f.n,...gemm(Tc,f.nope,f.kvLora,1/(H/tp)));
        const kp=f.kpool||1, nk=Math.min(S/2,f.topk);
        op("aidxp","indexer 投影 wq_i / wk_i / weights（各 TP rank 重复）","attn",f.n,...gemm(Tc,D,(f.qLora*f.idxH*f.idxD + D*f.idxD + D*f.idxH)/D,1));
        const fi=2*f.idxH*f.idxD*(S/2)/kp*Tc;
        op("aidx","indexer 打分 + top-k + KL loss（训练专有）","attn",f.n,{fl:fi*2,by:Tc*(S/2)/kp*4*2,pk:pkA},{fl:fi*2,by:Tc*(S/2)/kp*4*2,pk:pkA},null);
        const fl=2*(H/tp)*(lat+f.kvLora)*nk*Tc;
        op("acore","稀疏 absorbed MLA（top-"+f.topk+"）","attn",f.n,{fl,by:Tc*f.topk*4 + Tc*(H/tp)*(lat+f.kvLora)*2,pk:pkAt},{fl:2.5*fl,by:2*Tc*(H/tp)*(lat+f.kvLora)*2 + Tc*f.topk*4,pk:pkAt},null); coreFullOp=ops[ops.length-1];
        op("auv","W_uv","attn",f.n,...gemm(Tc,f.kvLora,f.vh,1/(H/tp)));
      } else {
        op("akvb","kv_b 上投影（非 absorbed，K 物化成每头）","attn",f.n,...gemm(Tc,f.kvLora,H*(f.nope+f.vh),tp));
        const fl=2*(H/tp)*((dq+f.vh)/2)*S*Tc;
        op("acore","causal MHA（qk "+dq+" / v "+f.vh+"）","attn",f.n,{fl,by:Tc*H*(2*dq+2*f.vh)*2/tp,pk:pkAt},{fl:2.5*fl,by:Tc*H*(4*dq+4*f.vh)*2/tp,pk:pkAt},null); coreFullOp=ops[ops.length-1];
      }
      if(f.gate){ op("agp","输出门 GEMM","attn",f.n,...gemm(Tc,D,H*f.vh,tp)); op("agate","sigmoid 输出门","attn",f.n,...elem(2*H*f.vh/tp,3*H*f.vh*2/tp,Tc,2)); }
      op("aout","o_proj GEMM","attn",f.n,...gemm(Tc,H*f.vh,D,tp));
    }
    tpBlocks.push({name:"全注意力",layers:f.n,inW: f.kind==="gqa"?D:(f.qLora+f.kvLora)});
  }

  /* ---- 线性注意力 ---- */
  if(l.n>0){
    const Hv=l.Hv/tp, lf=linFlops(l.kind,l.dk,l.dv), byTok=Hv*(l.kind==="KDA"?8200:5500);
    op("norm","pre-attn RMSNorm","misc",l.n,...elem(8*D/tp,3*D*2/tp,Tc,2));
    op("lproj","in_proj GEMM（q,k,v,门,β）","lin",l.n,...gemm(Tc,D,d.linIn,tp));
    op("lconv","短卷积 + SiLU + l2norm + 门","lin",l.n,...elem(2*l.conv*d.linConvDim/tp+20*Hv*l.dk,(d.linConvDim+2*Hv*l.dk)*2*2/tp,Tc,3));
    op("lcore",l.kind+" chunk kernel（"+(mg?"FLA":"attn_gym")+"，h 不保存、反向重算）","lin",l.n,{fl:Hv*lf.f*Tc,by:byTok*Tc,pk:pkA,k:5},{fl:Hv*lf.b*Tc,by:2.5*byTok*Tc,pk:pkA,k:8},null);
    op("lnorm","gated RMSNorm","lin",l.n,...elem(10*Hv*l.dv,3*Hv*l.dv*2,Tc,2));
    op("lout","out_proj GEMM","lin",l.n,...gemm(Tc,l.Hv*l.dv,D,tp));
    tpBlocks.push({name:"线性注意力",layers:l.n,inW:D});
  }

  /* ---- MoE ---- */
  const nM=d.nMoe, De=d.De, imb=c.imb||1;
  const dedup = ["deepep","hybridep","ncclep","distmoe"].includes(c.dispatcher);
  const perRankExp = e.E/ep;
  const kEff = dedup ? ep*Par.pDest(e.E, ep, e.k) : e.k;     // 去重后每 token 发往的不同 rank 数（含本 rank）
  const R = Tl*e.k*imb;                                     // 本 rank 上 expert 处理的行数（EP 组内均匀）
  op("norm","pre-MoE RMSNorm","misc",nM,...elem(8*D/tp,3*D*2/tp,Tc,2));
  op("router","router（"+(mg?"fp32 gating linear":"HiMidLo bf16 gate")+"）+ top-k + aux","moe",nM,{fl:2*Tl*D*e.E+10*Tl*e.E,by:D*e.E*4+Tl*(D*2+e.E*4*3),pk:pkA/2},{fl:4*Tl*D*e.E,by:D*e.E*8+Tl*e.E*4*3,pk:pkA/2},null);
  if(e.Dlat) op("latd","latent 下投影 D→"+e.Dlat,"moe",nM,...gemm(Tl,D,e.Dlat,1));
  op("permute","token permute（k 份拷贝"+(c.dispatcher==="distmoe"?"，融合在 Dist-MoE 里":"")+"）","moe",nM,{fl:0,by:Tl*De*2+R*De*2,pk:pkA},{fl:0,by:Tl*De*2+R*De*2,pk:pkA},null);
  op("moe1","grouped GEMM fc1（De→2F）","moe",nM,...gemm(R,De,2*e.F*etp,etp));
  op("act",(m.family==="kimi"?"SiTU-GLU":"SwiGLU")+" × routing prob","moe",nM,...elem(6*e.F,(2*e.F+e.F)*2,R,2));
  op("moe2","grouped GEMM fc2（F→De）","moe",nM,...gemm(R,e.F*etp,De,etp));
  op("unperm","unpermute（k 份求和）","moe",nM,{fl:R*De,by:R*De*2+Tl*De*2,pk:pkA},{fl:0,by:R*De*2+Tl*De*2,pk:pkA},null);
  if(e.Dlat) op("latu","latent norm + 上投影 "+e.Dlat+"→D","moe",nM,...gemm(Tl,e.Dlat,D,1));
  if(e.Fs>0){ op("shared","shared expert（TP 切）","moe",nM,...[0,1,2].map(i=>{ const a=gemm(Tc,D,3*e.Fs,tp)[i]; return a; }), {ovl: mg&&c.sharedOverlap}); tpBlocks.push({name:"shared expert",layers:nM,inW:D}); }
  if(m.dense.n>0){ op("norm","pre-MLP RMSNorm（dense 层）","misc",m.dense.n,...elem(8*D/tp,3*D*2/tp,Tc,2)); op("dense","dense FFN","moe",m.dense.n,...gemm(Tc,D,3*m.dense.F,tp)); tpBlocks.push({name:"dense FFN",layers:m.dense.n,inW:D}); }

  /* ---- 输出头：整段序列的 logits + CE（推理只需最后一个 token） ---- */
  const chunked = !mg;
  op("lm","lm_head（词表并行）","lm",1,...gemm(Tc,D,m.V,tp));
  op("ce",(mg?"vocab-parallel CE（fp32 logits，不分块）":"分块 CE（8 块）"),"lm",1,{fl:5*Tc*m.V/tp,by:Tc*m.V/tp*(2+4)*2,pk:pkA},{fl:3*Tc*m.V/tp,by:Tc*m.V/tp*(4+2),pk:pkA},null);
  if(tp>1) tpBlocks.push({name:"lm_head",layers:1,inW:D});

  /* ---- 通信（每 micro-batch，每 rank） ---- */
  if(mg && tp>1){
    for(const b of tpBlocks){ const msgAG=Tc*b.inW*2, msgRS=Tc*D*2;
      const tf=agT(msgAG,tp,lkTP,g)+agT(msgRS,tp,lkTP,g), tb=3*agT(Tc*D*2,tp,lkTP,g)+agT(msgRS,tp,lkTP,g);
      cm("tp","TP+SP："+b.name+"（fwd AG+RS，bwd 3AG+RS"+(rcFull?"，重算再 AG+RS":"")+"）","tp",b.layers,1,tf+tb+(rcFull?tf:0),(msgAG+msgRS)*(tp-1)/tp*(rcFull?6:5)/2,{ovl:c.tpOverlap?"comp":false}); }
    if(f.kind==="dsa") cm("tpidx","DSA indexer 的 x / q 额外 all-gather","tp",f.n,1,2*(agT(Tc*D*2,tp,lkTP,g)+agT(Tc*f.qLora*2,tp,lkTP,g)),Tc*(D+f.qLora)*2*(tp-1)/tp*2);
  } else if(!mg && tp>1){
    for(const b of tpBlocks){ const msg=Tc*D*2; const t1=2*agT(msg,tp,lkTP,g);
      cm("tp","TP："+b.name+"（Column AG + Row RS，fwd / bwd 各一次）","tp",b.layers,1,2*t1+(rcFull?t1:0),msg*(tp-1)/tp*4,{ovl:c.tpOverlap?"comp":false}); }
  }
  // CP：全注意力层走 Par.cpLayer（和“并行”页同一套公式）
  const cpMeth = mg ? ({p2p:"ring",allgather:"allgather",a2a:"ulysses"}[c.cpType]||"ring") : (Par.CP[c.cpType] ? c.cpType : "allgather");
  const cpLB = mg || c.cpLB!==false;                               // Megatron 的 CP 总是按 2C 段首尾配对
  const docs = T/S;                                                // 这一组 CP rank 一个 micro-batch 里的文档数
  const imbCP = (cp===1 || cpMeth==="ulysses" || cpLB) ? 1 : 2-1/cp;
  let cpFull=null, cpSwa=null;
  if(cp>1 && f.n>0){
    if(imbCP>1 && coreFullOp){ coreFullOp.fwd.t*=imbCP; coreFullOp.dgrad.t*=imbCP; coreFullOp.imb=imbCP; }
    const nSwa = f.kind==="gqa" ? (f.swa||0) : 0, nFull = f.n-nSwa;
    const per = (o2, n2) => o2 ? (o2.fwd.t + o2.dgrad.t + (rc("acore")?o2.fwd.t:0))/n2/docs : 0;   // 一层一篇的核心 F+B（含重算）
    const x = { S, C:cp, tp, m, d, hw, topo, g, ac:rcFull };
    cpFull = Par.cpLayer(cpMeth, Object.assign({ tCore:per(coreFullOp,nFull), kind:"full" }, x));
    if(!cpFull.ok) warn.push(cpFull.why);
    const how = { ring:"K / V 绕环 C−1 步，反向再传 K / V 和 dK / dV", allgather:"all-gather K / V，反向 reduce-scatter dK / dV", ulysses:"Q / K / V / O all-to-all（序列 ↔ 头）" }[cpMeth];
    cm("cpkv","CP "+Par.CP[cpMeth].name+"："+how+(f.kind==="dsa"?"（含 indexer key）":""),"cp",nFull,1,docs*cpFull.tw,docs*cpFull.wire,{ovl:"fixed",expT:nFull*docs*cpFull.exposed,lk:cpFull.lk,par:"cp"});
    if(nSwa){ cpSwa = Par.cpLayer(cpMeth, Object.assign({ tCore:per(coreSwaOp,nSwa), kind:"swa" }, x));
      cm("cpswa","CP 滑窗层："+(cpMeth==="ulysses"?"仍走两次全量 all-to-all":"只和邻居交换最后 "+f.W+" 个 token 的 K / V"),"cp",nSwa,1,docs*cpSwa.tw,docs*cpSwa.wire,{ovl:"fixed",expT:nSwa*docs*cpSwa.exposed,lk:cpSwa.lk,par:"cp"}); }
  }
  if(cp>1 && l.n>0){
    const Hv=l.Hv/tp;
    const head = l.kind==="GDN" || (mg ? c.linCP==="headwise" : c.linCP==="ulysses");
    if(head){ const sent=Tc*(d.linIn/tp + Hv*l.dv)*2; cm("cplin",l.kind+" CP headwise：in_proj 输出与 norm 输出 all-to-all（fwd 2 + bwd 2）","cp",l.n,1,2*a2aT(sent,cp,lkCP,g)+(rcFull||sel.has("gdn")?a2aT(sent,cp,lkCP,g):0),sent*(cp-1)/cp*2); }
    else { const msg=cp*Hv*l.dk*(l.dv+l.dk)*4; cm("cplin",l.kind+" CP chunkwise：fp32 段状态 + 转移矩阵 all-gather（fwd + bwd）+ 卷积 halo","cp",l.n,1,2*agT(msg,cp,lkCP,g),msg*(cp-1)/cp*2); }
  }
  // EP
  let a2aPerLayer=0, epL=null;
  if(ep>1 && !mg){
    const be = Par.EP[c.dispatcher] ? c.dispatcher : "standard";
    let tEF=0, tEB=0;
    for(const o of ops) if(["moe1","act","moe2"].includes(o.key)){ tEF+=o.fwd.t/nM; tEB+=(o.dgrad.t+o.wgrad.t+(rc(o.key)?o.fwd.t:0))/nM; }
    epL = Par.epLayer(be, { n:Tl, K:e.k, Z:e.E, E:ep, De, stride:1, hw, topo, g, ac:rcFull, prec:PR.f, imb, tExpF:tEF, tExpB:tEB, syncUs:c.hostSyncUs, mokF:c.mokF });
    if(!epL.ok) warn.push(epL.why);
    a2aPerLayer = epL.tw + epL.tAlpha;
    const ibShare = epL.wire>0 ? Math.min(1, epL.ib/epL.wire) : 0;
    cm("a2a","EP "+Par.EP[be].name+"：dispatch + combine ×"+epL.exch+(Par.EP[be].fused?"（融合在 expert GEMM 里）":""),"ep",nM,1,epL.tw+epL.tAlpha,epL.wire,{ovl:"fixed",expT:nM*(epL.extra-epL.tSync),lk:epL.lk,ibShare,par:"ep"});
    if(epL.syncs) cm("cnt","EP host sync："+(be==="deepep"?"do_cpu_sync 轮询计数":"D2H 拷 output_splits")+" ×"+epL.syncs,"ep",nM,1,epL.tSync,ep*e.E*8,{ovl:"fixed",expT:nM*epL.tSync,par:"ep"});
  } else if(ep>1){
    const nA = 4 + (rcA2A?2:0);
    const sent = Tl*kEff*De*2;
    const t1 = a2aT(sent,ep,lkEP,g, c.dispatcher==="hybridep"&&!lkEP.intra?undefined:undefined);
    a2aPerLayer=nA*t1;
    const name = {alltoall:"alltoall dispatcher（k 份拷贝）",deepep:"flex + DeepEP（按 rank 去重）",hybridep:"flex + HybridEP（去重，TMA/IBGDA）",ncclep:"flex + NCCL EP（MXFP8 线上）",standard:"标准 all-to-all（k 份拷贝 + host 同步）",distmoe:"Dist-MoE（对称显存，与 GEMM 融合）"}[c.dispatcher]||c.dispatcher;
    cm("a2a","EP "+name+"：dispatch + combine ×"+nA,"ep",nM,1,nA*t1*(c.dispatcher==="ncclep"?0.55:1),nA*sent*(ep-1)/ep,{ovl: c.dispatcher==="distmoe"?"moe" : (mg&&c.epOverlap?"comp":false)});
    if(!["hybridep","distmoe"].includes(c.dispatcher)) cm("cnt","EP 计数交换 + D2H 同步","ep",nM,1,(1+(rcA2A?1:0))*((lkEP.intra?g.alphaNv:g.alphaIb)*1e-6+c.hostSyncUs*1e-6),ep*e.E*8);
  }
  if(mg && etp>1){ const msg=R*etp*De*2; cm("etp","ETP：收到的行 all-gather + combine 前 reduce-scatter（fp32 probs 时 RS 翻倍）","ep",nM,1,2*(agT(msg,etp,lkTP,g)+agT(msg*2,etp,lkTP,g)),msg*3*(etp-1)/etp*2); }
  // PP p2p
  if(pp>1){ const msg=Tl*D*2; cm("pp","PP stage 间 p2p（fwd 激活 + bwd 梯度）","pp",1,1,2*(msg/(lkPP.bw*g.etaN)+ (lkPP.intra?g.alphaNv:g.alphaIb)*1e-6)*vpp,msg*2*vpp,{ovl:"comp",perStage:true}); }

  if(tokB>0) op("tok","按 token FP32 scale（单独的逐元素 op，未融进 GEMM epilogue）","misc",1,{fl:tokB/4,by:tokB,pk:pkA},null,null);

  /* ---- 汇总一个 micro-batch（整模型，按 rank） ---- */
  let tF=0,tD=0,tW=0,tR=0, flF=0,flB=0, moeT=0;
  for(const o of ops){ tF+=o.fwd.t; tD+=o.dgrad.t; tW+=o.wgrad.t; flF+=o.fwd.fl; flB+=o.dgrad.fl+o.wgrad.fl;
    if(rc(o.key)) { tR+=o.fwd.t; o.rc=true; }
    if(["moe1","moe2","act"].includes(o.key)) moeT+=o.fwd.t+o.dgrad.t+o.wgrad.t; }
  const attnT = ops.filter(o=>o.key==="acore").reduce((s,o)=>s+o.fwd.t+o.dgrad.t,0);
  const compMB = tF+tD+tW+tR;
  let exposed=0, hidden=0;
  for(const x of comm){ if(x.perStage) continue;
    const ex = x.ovl==="fixed" ? Math.min(x.t, x.expT) : x.ovl==="moe" ? Math.max(0,x.t-moeT) : x.ovl==="attn" ? Math.max(0,x.t-attnT)
             : x.ovl==="comp" ? x.t*0.15 : x.t;                                    // comp：userbuffers / combined 1F1B，经验上约 85% 可藏
    x.exp=ex; exposed+=ex; hidden+=x.t-ex; }
  if(mg && c.sharedOverlap){ const sh=ops.find(o=>o.key==="shared"); if(sh){ const save=Math.min(sh.fwd.t+sh.dgrad.t+sh.wgrad.t, a2aPerLayer*nM*0.5); exposed=Math.max(0,exposed-save); } }

  // FSDP2（torchtitan）：每层每个 micro-batch AG 参数（fwd，reshard 时 bwd 再一次）+ RS 梯度；HSDP 时分片组缩到 1/R，副本之间每步 all-reduce
  const Pdense = d.pDenseAll, Pexp = d.pExpertsAll;
  let fsdpMB=0, fsdp=null;
  if(!mg){
    const R = Math.max(1, c.hsdp||1), fs = dp*cp/R, fe = edp/R;
    if(!Number.isInteger(fs) || !Number.isInteger(fe)) warn.push("HSDP 副本数 "+R+" 必须同时整除 DP×CP = "+dp*cp+" 和 expert DP = "+edp);
    const reshard = c.reshard ?? (pp===1), aG = reshard?2:1, gB = c.gradDt==="bf16"?2:4;
    const lkD = Par.lk(hw,topo,fs,tp), lkE = Par.lk(hw,topo,fe,ep), lkR = Par.lk(hw,topo,R,fs*tp);
    const pRep = nM ? (d.pShared+d.pRouter+d.pLat)/tp : 0;           // MoE 层里随 dense 一起切的部分（shared、router、latent 投影）
    const pFull = f.n ? d.pAttn/tp + pRep : 0, pLin = l.n ? d.pLin/tp + pRep : 0, pDen = d.pDense/tp, pEH = (d.pEmb+d.pHead)/tp, pExpL = nM ? e.E*d.pExpert/ep : 0;
    const lay = [[pFull,f.n],[pLin,l.n],[pDen,m.dense.n],[pEH,1]];
    const tDense = lay.reduce((a,[P,n2])=>a+n2*Par.fsdpT(P,fs,lkD,g,aG,gB),0)/pp, bDense = lay.reduce((a,[P,n2])=>a+n2*Par.fsdpBytes(P,fs,aG,gB),0)/pp;
    const tExp = nM*Par.fsdpT(pExpL,fe,lkE,g,aG,gB)/pp, bExp = nM*Par.fsdpBytes(pExpL,fe,aG,gB)/pp;
    const firstAG = Par.wireT(Math.max(pFull,pLin)*(fs-1)/fs*2, lkD, g.etaAg, 1, g) + (fe>1 ? Par.wireT(pExpL*(fe-1)/fe*2, lkE, g.etaAg, 1, g) : 0);
    fsdpMB = tDense + tExp;
    fsdp = { R, fs, fe, aG, gB, reshard, lkD, lkE, lkR, pFull, pLin, pDen, pEH, pExpL, tDense, tExp, bDense, bExp, firstAG };
    comm.push({key:"fsdp",name:"FSDP2：参数 all-gather（fwd"+(reshard?" + bwd":"")+"）+ 梯度 reduce-scatter（"+(gB===4?"FP32":"BF16")+"），dense 组 "+fs+"、expert 组 "+fe+(R>1?"，HSDP "+R+" 个副本":""),cls:"dp",layers:1,passes:1,t:fsdpMB,bytes:bDense+bExp,ovl:"prefetch",perStage:true,par:"fsdp"});
  }

  const tMBstage = (compMB + exposed)/pp;
  // FSDP 在后台流上跑，但和 CP / EP 抢同一条链路：一个 micro-batch 至少要等最忙那条链路跑完；第一层的 gather 藏不住
  let fsdpExposedMB = 0;
  if(fsdp){
    const lane = { nvl:0, ib:0 }, put = (l2,t) => { if(t>0) lane[l2.intra?"nvl":"ib"] += t; };
    put(fsdp.lkD, fsdp.tDense); put(fsdp.lkE, fsdp.tExp);
    for(const x of comm) if(!x.perStage && x.lk){ if(x.ibShare!=null){ lane.ib += x.t*x.ibShare/pp; lane.nvl += x.t*(1-x.ibShare)/pp; } else put(x.lk, x.t/pp); }
    fsdp.lane = lane;
    fsdpExposedMB = Math.max(0, Math.max(lane.nvl, lane.ib) - tMBstage) + fsdp.firstAG;
  }
  const ppComm = comm.filter(x=>x.key==="pp").reduce((s,x)=>s+x.t,0)*0.1;
  const tMB = tMBstage + fsdpExposedMB + ppComm;

  /* ---- 步级：micro-batch 数、PP bubble、DP 同步、优化器 ---- */
  const microsPerStep = Math.max(1, Math.ceil(c.Gtok/(dp*T)));
  const bubble = pp>1 ? (pp-1)/vpp : 0;
  const tPipe = tMB*(microsPerStep + bubble);
  // 每 rank 的本地参数
  const PlD = Pdense/(tp*pp), PlE = Pexp/(ep*etp*pp);
  let tDP=0, tDPexp=0, dpNote="";
  if(mg){
    const dd=dp*cp;
    const rs = agT(PlD*4,dd,lkDP,g) + agT(PlE*4,edp,lkEDP,g), ag = agT(PlD*2,dd,lkDP,g)+agT(PlE*2,edp,lkEDP,g);
    tDP = rs+ag;
    const lastBwd = (tD+tW)/pp, firstFwd=tF/pp;
    tDPexp = (c.overlapGrad? Math.max(0,rs-lastBwd*0.9) : rs) + (c.overlapParam? Math.max(0,ag-firstFwd*0.9) : ag);
    dpNote="分布式优化器：梯度 reduce-scatter（fp32）+ 参数 all-gather（bf16），dense 组 DP×CP = "+dd+"，expert 组 EDP = "+edp;
  }
  const shardD = mg ? PlD/(dp*cp) : Pdense/(tp*pp)/fsdp.fs, shardE = mg ? PlE/edp : Pexp/(ep*pp)/fsdp.fe;
  const Pshard = shardD + shardE;
  let tOpt, optNote;
  if(c.offload){
    const host=hw.host*1e9;
    const xfer = Pshard*(4+2)/host;                                   // grad D2H + param H2D
    const cpu = Pshard*16*2/(c.cpuAdamGBs*1e9);                       // CPU Adam：读写 master + m + v
    tOpt = c.overlapOffload ? Math.max(xfer,cpu) : xfer+cpu;
    optNote="CPU Adam："+(c.overlapOffload?"D2H/H2D 与 CPU 计算重叠":"串行");
  } else { tOpt = Pshard*30/(hw.hbm*1e9*g.etaM) + Pshard*4/(hw.hbm*1e9*g.etaM); optNote="GPU fused Adam ≈ 30 B/参数 + 梯度范数读一遍"; }
  const tSmallAR = 4*((lkDP.intra?g.alphaNv:g.alphaIb)*1e-6)*2;       // grad norm、aux-free bias、loss finite 等标量 all-reduce
  let tAR = 0;
  if(fsdp && fsdp.R>1){ tAR = Par.wireT(2*(fsdp.R-1)/fsdp.R*Pshard*fsdp.gB, fsdp.lkR, g.etaAg, 1, g); tDP = tAR; tDPexp = Math.max(0, tAR - (tD+tW)/pp*0.9);
    dpNote = "HSDP：副本之间每步一次梯度 all-reduce（"+fsdp.R+" 个副本，"+fsdp.lkR.kind+"）"; }
  const stepT = tPipe + tDPexp + tOpt + tSmallAR;
  const tokens = microsPerStep*dp*T;
  const mfu = (flF+flB)/pp*microsPerStep/(stepT*pkF);

  /* ---- 显存 ---- */
  const L_st = m.L/pp;
  const actTok = {};      // 每 token 每层保存的激活字节（本 rank）
  const sp=tp;
  const attnAct = f.n>0 ? (f.kind==="gqa" ? (2*D*2 + (2*f.Hq+2*f.Hkv)*f.dh*2 + 4*f.Hq + (f.gate?2*f.Hq*f.dh*2:0))/sp
                 : (2*D*2 + (f.qLora+f.kvLora)*2 + (sel.has("mla_up_proj")?0:f.H*((f.nope+f.rope)*3+f.nope+f.vh)*2) + f.H*f.vh*2 + 4*f.H + (f.kind==="dsa"?f.topk*4*sp + (f.idxH*f.idxD+f.idxD)*2*sp:0))/sp) : 0;
  const linAct = l.n>0 ? (2*D*2 + d.linIn*2 + d.linConvDim*2 + l.Hv*(l.kind==="KDA"?1300:650) + l.Hv*l.dv*4*2 + l.Hv*l.dv*2)/sp : 0;
  const rowAct = e.k*imb*(De*2 + 2*e.F*2*(ttDef?0:1) + e.F*2*(sel.has("moe_act")?0:1) + 4);
  const moeAct = (2*D*2 + e.E*4*3)/sp + (sel.has("moe")?0:rowAct)/1 + (e.Dlat?(D+e.Dlat)*2:0) + (e.Fs>0&&!sel.has("shared_experts")?3*e.Fs*2/tp:0);
  let perTokLayers;
  if(rcFull){ perTokLayers = (m.L/pp)*D*2/sp; }
  else perTokLayers = (f.n*attnAct + l.n*(sel.has("gdn")?D*2/sp:linAct) + d.nMoe*moeAct + m.dense.n*(2*D*2+3*m.dense.F*2/tp))/pp;
  const oneLayerWS = Math.max(attnAct,linAct)+moeAct;          // 重算时一层的工作集
  const inflight = pp>1 ? Math.min(microsPerStep,pp)*(vpp>1?1+(pp-1)/(pp*vpp):1) : 1;
  const act = perTokLayers*Tc*inflight + (rcFull?oneLayerWS*Tc:0);
  const logits = Tc*m.V/tp*(mg?6:6/8);
  const a2aBuf = epL ? epL.buf : Tl*e.k*imb*De*2*2;
  const cpBuf = cp>1&&f.n>0 ? cp*Tc*d.kvTokElems*2 : 0;
  let states, paramsGPU, gathered=0;
  if(mg){
    paramsGPU = (PlD+PlE)*(2+4);                                      // bf16 参数 + fp32 main_grad
    states = c.offload ? 0 : shardD*12 + shardE*12;
  } else {
    paramsGPU = 0;
    states = c.offload ? 0 : Pshard*16;
    gathered = 2*(Pdense/tp/pp/(m.L/pp) + Pexp/ep/pp/(m.L/pp))*2*2;    // 当前 + 预取 block 的 bf16 参数与梯度
  }
  const work = c.workspaceGB*1e9;
  // 低精度权重缓存：前向一份；反向也低精度且权重是 1D 块时再从 BF16 量化一份列方向（2D 块共用一份）
  const qW = fq ? (mg ? PlD+PlE : Pdense/(tp*pp)+Pexp/(ep*pp))*(wbF + (bq&&!PR.w2d?wbB:0)) : 0;
  const used = paramsGPU + states + gathered + act + logits + a2aBuf + cpBuf + work + qW;
  const mem = { paramsGPU, states, gathered, act, logits, a2aBuf, cpBuf, work, qW, used, total:hw.mem*1e9, perTokLayers, inflight };

  return { ops, comm, tF, tD, tW, tR, compMB, exposed, hidden, tMB, tMBstage, fsdpMB, fsdpExposedMB, microsPerStep, bubble, tPipe, tDP, tDPexp, dpNote, tOpt, optNote, tSmallAR, stepT,
    tokens, tokPerGpu: tokens/(N*stepT), mfu, dp, edp, tp, cp, pp, vpp, ep, etp, T, Tc, Tl, R, kEff, Pshard, PlD, PlE, mem, warn, flF, flB, moeT, a2aPerLayer, nM,
    par:{ cpMeth, cpLB, imbCP, docs, cpFull, cpSwa, coreFullOp, ep:epL, fsdp, tAR } };
}

return { run, linFlops, PREC };
})();
