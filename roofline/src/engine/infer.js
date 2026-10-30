/* 推理核心：只描述 serving 引擎真正发出的 kernel 序列（paged KV、状态缓存、CUDA graph 补齐、采样、MTP 验证、DP-attention / EP 通信）。
   与训练核心不共享算子表。每条口径对应的仓库路径见 UI 的“口径与来源”。 */
const Infer = (function(){
const { dtB, peakOf, derive, qoW, link, agT, arT, a2aT, roof } = Common;

function range(a,b,s){ const r=[]; for(let x=a;x<=b;x+=s) r.push(x); return r; }
/* 引擎画像：来自三个引擎的默认值（research/inference_engines.md Part IV） */
const ENGINES = {
  sglang: { name:"SGLang", mixed:false, lmInGraph:true, samplerInGraph:false, specCap:48, defaultMoe:"tp",
    ladder:(hw,spec)=> spec ? [...range(1,8,1),...range(10,32,2),...range(40,64,4),...range(72,256,8)] : [1,2,4,8,12,...range(16,256,8),...range(272,496,16),512],
    maxGraph:(hw,spec)=> spec?48:512,
    arFusion:(m,c,T)=> (m.family==="qwen"||m.family==="kimi") && c.dpA===1 && T<=2048,
    stateDecode:()=>2, stateVerify:(m,N)=>N+3, prefillGraph:m=>m.family!=="kimi", prefillPad:0.06,
    hMat:(m,hw)=> !(m.lin.kind==="GDN" && hw.sm<100), mlaReexpand:true, sharedOverlap:"graph",
    kernels:{ gqa:"SM100 trtllm_mha（page 64）/ SM90 fa3", mla:"trtllm_mla", dsa:"dsa：SM100 trtllm 稀疏 MLA（fp8 KV）", gdn:"fused_recurrent_gated_delta_rule_packed_decode", kda:"fused KDA decode（每卡 3/6/12 头）", moe:"SM100 flashinfer_trtllm / SM90 Triton fused_moe；K3 flashinfer_mxfp4", ar:"flashinfer allreduce_fusion（Qwen）/ k3_ar_fusion（K3）/ 自研 custom AR" } },
  vllm: { name:"vLLM", mixed:true, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"ag",
    ladder:(hw,spec)=> [1,2,4,...range(8,248,8),...range(256, hw.sm>=100?1024:512,16)],
    maxGraph:(hw)=> hw.sm>=100?1024:512,
    arFusion:(m,c,T,D,hw)=> m.family==="qwen" && c.dpA===1 && T*D*2 <= (hw.sm>=100?1e6:5e5),
    stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>true, prefillPad:0.06,
    hMat:m=> m.lin.kind==="GDN", mlaReexpand:false, sharedOverlap:"small",
    kernels:{ gqa:"SM100 FlashInfer trtllm-gen / SM90 FA3", mla:"FLASHINFER_MLA（recipe：TOKENSPEED_MLA fp8）", dsa:"FLASHINFER_MLA_SPARSE + DeepGEMM paged MQA logits", gdn:"fused_recurrent_gated_delta_rule_packed_decode + causal_conv1d_update", kda:"_C.fused_kda_decode（纯 decode 批）/ FlashKDA（混合批）", moe:"SM100 TRTLLM-gen / SM90 Triton 或 CUTLASS", ar:"Qwen：Inductor AR+RMSNorm 融合；K3/GLM：独立 flashinfer AR + norm" } },
  tokenspeed: { name:"TokenSpeed", mixed:false, lmInGraph:true, samplerInGraph:true, specCap:Infinity, defaultMoe:"replicated",
    ladder:(hw,spec)=> [1,2,4,...range(8, spec?80:160, 8)],
    maxGraph:(hw,spec)=> spec?80:160,
    arFusion:(m,c,T)=> m.family!=="glm" && c.dpA===1 && T<=2048,
    stateDecode:m=> m.lin.kind==="KDA"?3:2, stateVerify:()=>3, prefillGraph:m=>m.full.kind!=="dsa", prefillPad:0.125,
    hMat:()=>false, mlaReexpand:true, sharedOverlap:"graph",
    kernels:{ gqa:"triton_gqa_prologue + trtllm-gen / fa3", mla:"tokenspeed_mla（split-KV + reduce）", dsa:"trtllm-gen NoPE 稀疏 MLA（prefill 也走 absorbed）", gdn:"triton_causal_conv1d_update + flashinfer_gdn_decode_step", kda:"verify + 批量 replay commit（单 token 也走）", moe:"flashinfer_trtllm（Blackwell）/ DeepGEMM masked（DeepEP）", ar:"TRT-LLM Lamport allreduce_residual_rmsnorm，MoE 的 AR 推迟到下一层" } },
  titan: { name:"TorchTitan TitanRL", mixed:true, lmInGraph:false, samplerInGraph:false, specCap:Infinity, defaultMoe:"standard",
    ladder:(hw)=> [1,2,4,...range(8,248,8),...range(256,512,16)], maxGraph:()=>512,
    arFusion:()=>false, stateDecode:()=>2, stateVerify:(m,N)=>N+1, prefillGraph:()=>false, prefillPad:0,
    hMat:()=>true, mlaReexpand:false, sharedOverlap:"none",
    kernels:{ gqa:"varlen（FA3 / FA4 / cuDNN）", mla:"torchtitan MLA（非 absorbed）", dsa:"无原生 GLM-5 模型", gdn:"attn_gym GDN", kda:"attn_gym KDA", moe:"torch._grouped_mm / Dist-MoE", ar:"DTensor all-reduce（无融合）" } }
};

const padTo = (ladder, x) => { for(const v of ladder) if(v>=x) return v; return x; };
const cover = (k,E,T) => 1-Math.pow(1-k/E, Math.max(0,T));

/* 一次 forward（decode 或 prefill chunk）的逐算子时间 */
function step(m, hw, topo, c, g){
  const d=derive(m), P=ENGINES[c.engine], f=m.full, l=m.lin, e=m.moe, D=m.D;
  const N=c.N, dpA=Math.max(1,c.dpA), tpA=N/dpA, dec=c.phase==="decode";
  const wbE=dtB(c.wDt), wbD=dtB(c.dDt), kvb=dtB(c.kvDt), sb=c.stateDt==="bf16"?2:4;
  const pkD=peakOf(hw,c.dAct||c.dDt), pkE=peakOf(hw,c.eAct||c.wDt), pkA=peakOf(hw,"bf16");   // eAct / dAct：expert / 其余线性层的激活精度（W4A16 / W4A8 时 MMA 跑在激活的峰值上）
  const slow=(c.bi?c.biFactor:1)/(c.implEff||1);
  const specK = dec ? (c.specK||0) : 0, w = specK>0 ? specK+1 : 1;
  const warn=[];
  // 批大小与 CUDA graph 补齐
  let Breq = dec ? c.B : 1;
  if(dec && specK>0 && Breq>P.specCap){ warn.push(P.name+" 开投机解码时 max_running_requests 被限制为 "+P.specCap); Breq=P.specCap; }
  const ladder=P.ladder(hw,specK>0), maxG=P.maxGraph(hw,specK>0);
  let graph, rowsG, rowsR;
  if(dec){
    const units = c.engine==="vllm"||c.engine==="titan" ? Breq*w : Breq;
    graph = c.graph && units<=maxG;
    const unitsPad = graph ? padTo(ladder, units) : units;
    rowsG = c.engine==="vllm"||c.engine==="titan" ? unitsPad : unitsPad*w;
    rowsR = Breq*w;
  } else {
    graph = c.graph && P.prefillGraph(m) && c.C<=8192;
    rowsR = c.C; rowsG = graph ? Math.ceil(c.C*(1+P.prefillPad*0.5)) : c.C;
  }
  const Bp = dec ? rowsG/w : 1;   // 补齐后的请求数（只影响随行数缩放的算子）
  const ops=[], comm=[];
  const op=(key,name,group,layers,flops,bytes,peak,kern,extra)=>{ if(layers<=0) return;
    const r=roof(flops,bytes,peak,hw,g,kern); r.t*=slow; r.tc*=slow; r.tm*=slow;
    ops.push(Object.assign({key,name,group,layers,flops:flops*layers,bytes:bytes*layers,t:r.t*layers,tc:r.tc*layers,tm:r.tm*layers,tf:r.tf*layers,bound:r.bound,I:r.I,kernels:(kern||0)*layers},extra||{})); };
  const cm=(key,name,layers,t,bytes,extra)=>{ if(layers<=0||t<=0) return; comm.push(Object.assign({key,name,layers,t:t*layers,bytes:bytes*layers},extra||{})); };
  const lkA=link(hw,topo,tpA,1), lkN=link(hw,topo,N,1);
  const ctx = dec ? c.n : c.p;              // decode：已有上下文；prefill：本 chunk 之前的前缀

  /* ---- 层边界：AR(+add+RMSNorm) ---- */
  const boundaries = m.L*2;
  const fused = tpA>1 && P.arFusion(m,c,rowsG,D,hw);
  const arAlgo = rowsG*D*2 <= 512*1024 ? "oneshot" : "twoshot";
  if(tpA>1){
    cm("ar", fused?"AR + 残差 + RMSNorm（融合）":"TP all-reduce", boundaries, arT(rowsG*D*2,tpA,lkA,g,arAlgo), rowsG*D*2*(tpA-1)/tpA);
    if(!fused) op("norm","残差 + RMSNorm","misc",boundaries,8*rowsG*D,3*rowsG*D*2,pkA,1);
  } else op("norm","残差 + RMSNorm","misc",boundaries,8*rowsG*D,3*rowsG*D*2,pkA,1);
  if(m.mhc) op("mhc","mHC 4 路残差混合（pre / post）","misc",m.L*2,4*m.mhc*rowsG*D,2*m.mhc*rowsG*D*2,pkA,2);
  if(m.attnRes) op("attnres","AttnRes 块残差（读历史快照）","misc",m.L,2*(m.attnRes/2+1)*rowsG*D,(m.attnRes/2+1)*rowsG*D*2,pkA,1);

  /* ---- 全注意力层 ---- */
  let kvTokRank=0, idxTok=0;
  if(f.n>0){
    if(f.kind==="gqa"){
      const qo=qoW(f), kvo=2*f.Hkv*f.dh, Hl=f.Hq/tpA, kvHeads=Math.max(1,f.Hkv/tpA);
      const nSwa=f.swa||0, nGlob=f.n-nSwa;
      kvTokRank = 2*kvHeads*f.dh*kvb;
      op("aproj","QKV"+(f.gate?"（含输出门）":"")+" GEMM","attn",f.n,2*rowsG*D*(qo+kvo)/tpA, D*(qo+kvo)/tpA*wbD + rowsG*(D+(qo+kvo)/tpA)*2, pkD,1);
      op("aprol","QK-norm + RoPE + KV 写入","attn",f.n,10*rowsG*(Hl+2*kvHeads)*f.dh, rowsG*(Hl+2*kvHeads)*f.dh*2*2 + rowsR*kvTokRank, pkA, c.engine==="sglang"&&f.gate?2:1);
      // 滑窗层只读最近 W 个 token 的 KV；全局层读整段
      const core=(layers,keyCap,tag)=>{ const cx=Math.min(ctx,keyCap);
        if(dec) op("acore","paged attention decode（"+P.kernels.gqa+"）"+tag,"attn",layers, 4*Hl*f.dh*cx*Breq*w, Breq*cx*kvTokRank + rowsR*Hl*f.dh*2*2, pkA,1,{scalesWith:"real"});
        else op("acore","causal attention（prefill / extend）"+tag,"attn",layers, 4*Hl*f.dh*rowsR*Math.min(ctx+rowsR/2,keyCap), (cx+rowsR)*kvTokRank + rowsR*Hl*f.dh*2*2, pkA,1); };
      core(nGlob, Infinity, nSwa?"：全局层":"");
      if(nSwa) core(nSwa, f.W, "：滑窗 "+f.W);
      if(f.gate) op("agate",f.gateHead?"每头 sigmoid 门":"sigmoid 输出门","attn",f.n,2*rowsG*Hl*f.dh,3*rowsG*Hl*f.dh*2,pkA,1);
      op("aout","o_proj GEMM","attn",f.n,2*rowsG*f.Hq*f.dh*D/tpA, f.Hq*f.dh*D/tpA*wbD + rowsG*(f.Hq*f.dh/tpA+D)*2, pkD,1);
    } else {
      const H=f.H, Hl=H/tpA, dq=f.nope+f.rope, lat=f.kvLora+f.rope, dcp=Math.max(1,c.dcp||1);
      kvTokRank = lat*kvb/dcp;                                    // latent 无头维度，TP 切不开，只能靠 DCP
      op("aqa","fused_qkv_a GEMM（D→q_lora+kv_lora+rope，各卡复制）","attn",f.n,2*rowsG*D*(f.qLora+lat), D*(f.qLora+lat)*wbD + rowsG*(D+f.qLora+lat)*2, pkD,1);
      op("anorm","q_a / kv_a RMSNorm + KV 写入","attn",f.n,8*rowsG*(f.qLora+f.kvLora), rowsG*(f.qLora+f.kvLora)*2*2 + rowsR*kvTokRank*dcp/dcp, pkA,2);
      op("aqb","q_b GEMM","attn",f.n,2*rowsG*f.qLora*Hl*dq, f.qLora*Hl*dq*wbD + rowsG*(f.qLora+Hl*dq)*2, pkD,1);
      if(f.gate) op("agp","输出门 GEMM（D→H·v，TP 切）","attn",f.n,2*rowsG*D*Hl*f.vh, D*Hl*f.vh*wbD, pkD,1);
      if(f.kind==="dsa"){
        const ip=f.idxH*f.idxD+f.idxD+f.idxH, kp=f.kpool||1;
        idxTok = (f.idxD+4)/kp;                                       // fp8 index-K + scale，按池存
        op("aidxp","indexer 投影 wq_i / wk_i / weights（各卡复制）","attn",f.n,2*rowsG*(f.qLora*f.idxH*f.idxD + D*f.idxD + D*f.idxH), (f.qLora*f.idxH*f.idxD+D*f.idxD+D*f.idxH)*wbD + rowsG*ip*2, pkD,2);
        const nk = dec ? ctx : ctx+rowsR/2;
        if(dec ? ctx>f.topk : ctx+rowsR>f.topk)
          op("aidx","lightning indexer 打分 + top-k（n/"+kp+" 个池）","attn",f.n, 2*f.idxH*f.idxD*(nk/kp)*rowsR, (dec?Breq*ctx:(ctx+rowsR))*idxTok + 2*4*rowsR*nk/kp, peakOf(hw,"fp8"),2,{scalesWith:dec?"real":undefined});
      }
      // absorb：q 乘 W_uk 到 latent，attention 在 latent 上做 MQA，再乘 W_uv
      const absorb = dec || f.kind==="dsa" && c.engine==="tokenspeed";
      if(absorb){
        op("auk","W_uk absorb bmm","attn",f.n,2*rowsG*Hl*f.nope*f.kvLora, Hl*f.nope*f.kvLora*2 + rowsG*Hl*(f.nope+f.kvLora)*2, pkA,1);
        const nKV = f.kind==="dsa" ? Math.min(dec?ctx:ctx+rowsR/2, f.topk+3) : (dec?ctx:ctx+rowsR/2);
        const tag = f.kind==="dsa" ? "稀疏 MLA（≤ top-k 个 latent，"+P.kernels.dsa+"）" : "MLA decode（"+P.kernels.mla+"）";
        if(dec) op("acore",tag,"attn",f.n, 2*Hl*(lat+f.kvLora)*nKV*Breq*w, Breq*nKV*kvTokRank*(f.kind==="dsa"?1:1) + rowsR*Hl*(lat+f.kvLora)*2, pkA,2,{scalesWith:"real"});
        else op("acore",tag,"attn",f.n, 2*Hl*(lat+f.kvLora)*nKV*rowsR, (ctx+rowsR)*kvTokRank, pkA,2);
        op("auv","W_uv bmm","attn",f.n,2*rowsG*Hl*f.kvLora*f.vh, Hl*f.kvLora*f.vh*2 + rowsG*Hl*(f.kvLora+f.vh)*2, pkA,1);
      } else {
        // prefill：MHA 形式，kv_b 把新 token（以及 SGLang / TokenSpeed 下的整个前缀）展开成每头 K/V
        const reexp = P.mlaReexpand && ctx>0;
        const expTok = rowsR + (reexp?ctx:0);
        op("akvb","kv_b 上投影"+(reexp?"（含前缀重展开）":""),"attn",f.n,2*expTok*f.kvLora*Hl*(f.nope+f.vh), f.kvLora*Hl*(f.nope+f.vh)*wbD + expTok*(f.kvLora + Hl*(f.nope+f.vh))*2, pkD,1);
        const nKV = f.kind==="dsa" && ctx+rowsR>f.topk ? Math.min(ctx+rowsR/2,f.topk) : ctx+rowsR/2;
        op("acore",f.kind==="dsa"&&ctx+rowsR>f.topk?"稀疏 MHA prefill（top-k）":"causal MHA prefill（qk "+dq+" / v "+f.vh+"）","attn",f.n, 2*Hl*(dq+f.vh)*rowsR*nKV, expTok*Hl*(dq+f.vh)*2 + rowsR*Hl*(dq+f.vh)*2, pkA,1);
      }
      if(f.gate) op("agate","sigmoid 输出门","attn",f.n,2*rowsG*Hl*f.vh,3*rowsG*Hl*f.vh*2,pkA,1);
      op("aout","o_proj GEMM","attn",f.n,2*rowsG*H*f.vh*D/tpA, H*f.vh*D/tpA*wbD + rowsG*(Hl*f.vh+D)*2, pkD,1);
    }
  }

  /* ---- 线性注意力层（GDN / KDA） ---- */
  let Sl=0, Cl=0;
  if(l.n>0){
    const Hv=l.Hv/tpA; Sl = Hv*l.dk*l.dv*sb; Cl = d.linConvDim/tpA*(l.conv-1)*2;
    op("lproj","in_proj GEMM（q,k,v,门,β）","lin",l.n,2*rowsG*D*d.linIn/tpA, D*d.linIn/tpA*wbD + rowsG*(D+d.linIn/tpA)*2, pkD, l.kind==="KDA"?2:1);
    if(dec){
      const mult = specK>0 ? P.stateVerify(m,w) : P.stateDecode(m);
      op("lconv","短卷积 state 更新","lin",l.n,2*l.conv*rowsR*d.linConvDim/tpA, Breq*Cl*2 + rowsR*d.linConvDim/tpA*2*2, pkA,1,{scalesWith:"real"});
      op("lcore",l.kind+" 递推状态更新（"+(l.kind==="KDA"?P.kernels.kda:P.kernels.gdn)+"）","lin",l.n, 8*Hv*l.dk*l.dv*rowsR, mult*Breq*Sl + rowsR*Hv*(2*l.dk+2*l.dv)*2, pkA, c.engine==="tokenspeed"&&l.kind==="KDA"?2:1,{scalesWith:"real",stateMult:mult});
    } else {
      const C=64, hm=P.hMat(m,hw);
      op("lconv","短卷积（整段）","lin",l.n,2*l.conv*rowsR*d.linConvDim/tpA, rowsR*d.linConvDim/tpA*2*2, pkA,1);
      op("lcore",l.kind+" chunk kernel（C=64"+(hm?"，h 写回 HBM":"，融合不落 h")+"）","lin",l.n,
         rowsR*Hv*(6*l.dk*l.dv + C*(6*l.dk+4*l.dv) + C*C/3), rowsR*Hv*(2*l.dk+2*l.dv+l.dk)*2*2 + (hm? 2*rowsR*Hv*l.dk*l.dv*2/C : 0) + 2*Sl, pkA, hm?6:3);
    }
    op("lnorm","gated RMSNorm","lin",l.n,10*rowsG*Hv*l.dv,3*rowsG*Hv*l.dv*2,pkA,1);
    op("lout","out_proj GEMM","lin",l.n,2*rowsG*l.Hv*l.dv*D/tpA, l.Hv*l.dv*D/tpA*wbD + rowsG*(Hv*l.dv+D)*2, pkD,1);
  }

  /* ---- MoE 层 ---- */
  const nMoe=d.nMoe, De=d.De, moePar=c.moePar||P.defaultMoe;
  const Tg = rowsG*dpA;                     // MoE 一侧看到的全局 token（含补齐行）
  const Tloc = rowsG;
  const replicatedIn = moePar==="replicated" || moePar==="tp";
  const routerT = moePar==="replicated" ? Tg : Tloc;
  op("router","router GEMM + top-k（各卡复制）","moe",nMoe,2*routerT*D*e.E + 8*routerT*e.E, D*e.E*2 + routerT*(D*2+e.E*4), pkA, 2);
  if(e.Dlat) op("latd","latent 下投影 D→"+e.Dlat+"（各卡复制）","moe",nMoe,2*Tloc*D*e.Dlat, D*e.Dlat*2 + Tloc*(D+e.Dlat)*2, pkA,1);
  const Eloc=(e.E+(c.redundant||0))/N, cv=cover(e.k,e.E,Tg);
  const touched = moePar==="tp" ? e.E*cv : Eloc*cv;   // tp：每卡持有全部 expert 的 F/N 切片
  const wSlice = moePar==="tp" ? 3*De*e.F/N : 3*De*e.F;
  const rowsExp = Tg*e.k/N*(c.imb||1);
  const eA = c.eAct || (c.wDt==="bf16"?"bf16":c.wDt);
  if(c.qSep && eA!=="bf16") op("moeq","expert 输入激活量化（单独 kernel）","moe",nMoe,4*Tloc*De, Tloc*De*(2+dtB(eA)), pkA,1);
  op("moe","routed experts（"+P.kernels.moe+"）","moe",nMoe, 2*rowsExp*3*De*e.F/(moePar==="tp"?1:1), touched*wSlice*wbE + rowsExp*(De*2+3*e.F*2/(moePar==="tp"?N:1)), pkE, 3,{touched,cover:cv});
  if(e.Fs>0){ const Fl=e.Fs/tpA;
    op("shared","shared expert"+(e.sharedGate?" + sigmoid 门":""),"moe",nMoe,6*Tloc*D*Fl, 3*D*Fl*wbD + Tloc*(D+3*Fl)*2, pkD,2,
      {overlap: (P.sharedOverlap==="graph"&&graph&&dec) || (P.sharedOverlap==="small"&&Tloc<=256) ? "moe" : false}); }
  if(e.Dlat){
    if(tpA>1) cm("latar","latent 宽度 all-reduce（"+e.Dlat+"）",nMoe,arT(Tloc*e.Dlat*2,tpA,lkA,g,"oneshot"),Tloc*e.Dlat*2*(tpA-1)/tpA);
    op("latu","latent RMSNorm + 上投影 "+e.Dlat+"→D（各卡复制）","moe",nMoe,2*Tloc*e.Dlat*D, e.Dlat*D*2 + Tloc*(D+e.Dlat)*2, pkA,2);
  }
  if(m.dense.n>0) op("dense","dense FFN","moe",m.dense.n,6*rowsG*D*m.dense.F/tpA, 3*D*m.dense.F/tpA*wbD + rowsG*(D+3*m.dense.F/tpA)*2, pkD,2);
  // MoE 通信
  const lkE = link(hw,topo,N,1);
  if(N>1){
    if(moePar==="tp" || moePar==="replicated"){
      if(dpA>1){ cm("dpg","DP-attention：all-gather 进 MoE",nMoe,agT(Tg*De*2,dpA,lkE,g),Tg*De*2*(dpA-1)/dpA); cm("dps","DP-attention：reduce-scatter 回各 rank",nMoe,agT(Tg*De*2,dpA,lkE,g),Tg*De*2*(dpA-1)/dpA); }
      else if(!e.Dlat) {/* MoE 结果的 AR 并进层边界的 AR（已计入） */}
    } else if(moePar==="ag"){
      cm("dpg","EP all-gather（allgather_reducescatter）",nMoe,agT(Tg*De*2,N,lkE,g),Tg*De*2*(N-1)/N);
      cm("dps","EP reduce-scatter",nMoe,agT(Tg*De*2,N,lkE,g),Tg*De*2*(N-1)/N);
    } else {
      const sentD = Tloc*e.k*De*(moePar==="deepep"&&c.dispFp8 ? 1+4/128 : 2), sentC = Tloc*e.k*De*2;
      const al = moePar==="deepep" ? (dec?g.a2aLLus:g.alphaNv) : undefined;
      const fusedA2A = moePar==="fused";
      const tag = moePar==="deepep" ? (dec?"DeepEP low-latency":"DeepEP normal") : moePar==="standard" ? "torch all-to-all（k 份拷贝）" : moePar==="hybridep" ? "HybridEP" : "融合 a2a（megamoe / Dist-MoE / MNNVL）";
      cm("disp",tag+" dispatch",nMoe,a2aT(sentD,N,lkE,g,al),sentD*(N-1)/N,{overlap:fusedA2A?"moe":(c.tbo?"tbo":false)});
      cm("comb",tag+" combine",nMoe,a2aT(sentC,N,lkE,g,al),sentC*(N-1)/N,{overlap:fusedA2A?"moe":(c.tbo?"tbo":false)});
      if(moePar==="standard") cm("cnt","token 计数交换 + D2H 同步",nMoe,(lkE.intra?g.alphaNv:g.alphaIb)*1e-6 + (c.hostSyncUs||20)*1e-6, N*e.E*8/N);
    }
  }

  /* ---- 每步一次：lm_head、logits 收集、采样 ---- */
  const lmRows = dec ? (P.lmInGraph&&graph ? rowsG : rowsR) : Math.max(1,c.reqPerChunk||1);
  op("lm","lm_head"+(dec&&P.lmInGraph&&graph?"（在 graph 里，按补齐行）":""),"lm",1,2*lmRows*D*m.V/tpA, D*m.V/tpA*wbD + lmRows*m.V/tpA*4, pkD,1);
  if(tpA>1) cm("lmg","logits all-gather（词表并行）",1,agT(lmRows*m.V*4,tpA,lkA,g),lmRows*m.V*4*(tpA-1)/tpA);
  const sampRows = dec ? (P.samplerInGraph&&graph ? rowsG : rowsR) : lmRows;
  op("samp","采样（top-k/top-p，"+(P.samplerInGraph?"在 graph 里":"eager")+"）","lm",1,5*sampRows*m.V,4*sampRows*m.V*4,pkA,3);

  /* ---- 汇总 ---- */
  const moeT = ops.filter(o=>o.key==="moe").reduce((s,o)=>s+o.t,0);
  let tOps=0, tHidden=0;
  for(const o of ops){ if(o.overlap==="moe"){ tHidden+=o.t; } else tOps+=o.t; }
  const sharedT = ops.filter(o=>o.overlap==="moe").reduce((s,o)=>s+o.t,0);
  let tComm=0, fusedComm=0, tboComm=0;
  for(const x of comm){ if(x.overlap==="moe") fusedComm+=x.t; else if(x.overlap==="tbo") tboComm+=x.t; else tComm+=x.t; }
  const exposedOverlap = Math.max(0, sharedT + fusedComm - moeT);  // 共享 expert 与融合 a2a 都藏在 routed expert 后面
  const tComp = tOps;
  let gpu = tComp + tComm + exposedOverlap + (c.tbo ? Math.max(0,tboComm - tComp*0.5) : tboComm);
  const nKern = ops.reduce((s,o)=>s+o.kernels,0) + comm.length*1;
  const cpuLaunch = graph ? g.graphLaunchUs*1e-6 : nKern*g.launchUs*1e-6;
  const dpSync = (c.engine==="vllm"||c.engine==="titan") && dpA>1 ? g.dpSyncUs*1e-6 : 0;
  const stepT = Math.max(gpu + (graph?g.graphLaunchUs*1e-6:0), cpuLaunch, g.schedUs*1e-6) + dpSync;
  return { ops, comm, tComp, tComm, exposedOverlap, gpu, stepT, graph, rowsG, rowsR, Breq, Bp, w, specK, tpA, dpA, N, Tg, kvTokRank, idxTok, Sl, Cl, nKern, cpuLaunch, fused, warn, moePar, P };
}

/* 投机解码的 draft 一项，按方法分开算。MTP：目标模型一层的平均代价（第一步按验证宽度，之后每步 1 行）+ 每步一次 lm_head。
   另外三种按各自的结构：EAGLE-3 每步一层 dense + 裁剪词表的 lm_head；DFlash 一个约 5 层的小模型对 B·(k+1) 行做一次 forward，
   上下文 KV 由目标模型的特征投影出来；DSpark 同 DFlash（B·k 行），外加 k 次 [B, r] × [r, V] 的 Markov 偏置。draft 层按 dense（≈ 12·D² 参数）、8 个 KV 头 × 128 算。 */
const SPEC = { mtp:{ name:"MTP" }, eagle3:{ name:"EAGLE-3", layers:1, vocab:32000 }, dflash:{ name:"DFlash", layers:5 }, dspark:{ name:"DSpark", layers:5, rank:256 } };
const DRAFT_KVW = 1024;
function draftCost(m, hw, c, g, r, base1){
  const k=r.specK, B=r.Breq, D=m.D, tp=r.tpA, S=SPEC[c.specMethod]?c.specMethod:"mtp";
  if(S==="mtp"){ const perLayerW=(r.gpu-opT(r,"lm")-opT(r,"samp"))/m.L, perLayer1=(base1.gpu-opT(base1,"lm")-opT(base1,"samp"))/m.L;
    return { t: perLayerW + (k-1)*perLayer1 + k*opT(base1,"lm") }; }
  const pk=peakOf(hw,"bf16"), R=(fl,by,kern)=>roof(fl,by,pk,hw,g,kern||1).t, n=c.n, Wl=12*D*D, kvw=DRAFT_KVW;
  const arc=base1.comm.filter(x=>x.key==="ar"), ar1=tp>1&&arc.length ? arc.reduce((s,x)=>s+x.t,0)/arc.reduce((s,x)=>s+x.layers,0) : 0;
  const layer=rows=>R(2*rows*Wl/tp, Wl*2/tp + rows*D*2*6, 6) + R(4*rows*n*D/tp, B*n*2*kvw*2/tp) + 2*ar1;      // 一层 dense + 读这一层的上下文 KV
  const lm=(rows,V)=>R(2*rows*D*V/tp, D*V*2/tp + rows*V*4/tp);
  if(S==="eagle3"){ const w=k+1, Vd=SPEC.eagle3.vocab;
    const t=R(2*B*w*3*D*D/tp, 3*D*D*2/tp) + layer(B*w) + (k-1)*layer(B) + k*lm(B,Vd);
    return { t, rows:B*w }; }
  const L=SPEC[S].layers, rows=B*(S==="dspark"?k:k+1), ctxRows=B*(k+1);
  const ctx=R(2*ctxRows*5*D*D/tp, 5*D*D*2/tp) + R(2*ctxRows*D*L*2*kvw/tp, D*L*2*kvw*2/tp);   // 目标特征 → 每层上下文 K / V
  let t=ctx + L*layer(rows) + lm(B*k,m.V);
  if(S==="dspark"){ const rk=SPEC.dspark.rank; t+=k*R(2*B*rk*m.V, rk*m.V*2 + B*m.V*4); }
  return { t, rows };
}

/* 投机解码：verify 步 + draft + 提交 */
function decode(m, hw, topo, c, g){
  const r = step(m,hw,topo,c,g);
  r.draftT=0; r.commitT=0;
  if(r.specK>0){
    const base1 = step(m,hw,topo,Object.assign({},c,{specK:0}),g);
    r.draftT = draftCost(m,hw,c,g,r,base1).t;
    if(m.lin.n>0) r.commitT = c.engine==="tokenspeed" ? g.launchUs*1e-6*2 + 2*r.Breq*r.Sl*m.lin.n/(hw.hbm*1e9*g.etaM) : 2*r.Breq*r.Sl*m.lin.n/(hw.hbm*1e9*g.etaM);
    r.stepT += r.draftT + r.commitT;
  }
  r.tokPerStep = r.Breq*(1+(r.specK>0?c.accept:0));
  r.tpot = r.stepT/(1+(r.specK>0?c.accept:0));
  r.tokPerGpu = r.tokPerStep*r.dpA/r.N/r.stepT;
  r.mem = memory(m,hw,c,r);
  return r;
}
const opT=(r,k)=>r.ops.filter(o=>o.key===k).reduce((s,o)=>s+o.t,0);

/* prefill：按 chunk 迭代一个长 prompt，得到 TTFT；单 chunk 的吞吐 */
function prefill(m, hw, topo, c, g){
  const C=c.C, P=c.P||C, chunks=[]; let t=0;
  for(let p=0;p<P;p+=C){ const cc=Math.min(C,P-p); const r=step(m,hw,topo,Object.assign({},c,{phase:"prefill",C:cc,p}),g); chunks.push({p,C:cc,t:r.stepT}); t+=r.stepT; if(chunks.length>256) break; }
  const r = step(m,hw,topo,Object.assign({},c,{phase:"prefill",p:c.p||0}),g);
  r.ttft=t; r.chunks=chunks; r.tokPerGpu = c.C*r.dpA/r.N/r.stepT; r.mem=memory(m,hw,Object.assign({},c,{phase:"decode",B:1}),r);
  return r;
}

/* 显存：权重 + KV 池 + 线性注意力状态槽 + 预留 */
function memory(m, hw, c, r){
  const d=derive(m), f=m.full, l=m.lin, e=m.moe, N=c.N, tpA=N/Math.max(1,c.dpA);
  const wbE=dtB(c.wDt), wbD=dtB(c.dDt), kvb=dtB(c.kvDt), sb=c.stateDt==="bf16"?2:4, P=ENGINES[c.engine];
  const wAttn = f.n*(((d.pQKV+d.pO)/tpA) + d.pAttnRep + d.pIdx)*wbD;
  const wLin = l.n*d.pLin/tpA*wbD;
  const wExp = e.E ? d.nMoe*d.pRouted*(e.E+(c.redundant||0))/e.E/N*wbE : 0;
  const wMoeRep = d.nMoe*((d.pShared)/tpA + d.pRouter + d.pLat)*wbD;
  const wDense = m.dense.n*d.pDense/tpA*wbD;
  const wEmb = 2*m.V*m.D/tpA*wbD;
  const sm = c.specK>0 && SPEC[c.specMethod] && c.specMethod!=="mtp" ? SPEC[c.specMethod] : null;   // 外挂 drafter：自己的权重和上下文 KV
  const wMtp = (c.specK>0 && m.mtp && !sm) ? (d.total-d.pEmb-d.pHead)/m.L/N*Math.min(wbD,wbE)*1.0 : 0;
  const wDraft = sm ? (sm.layers*12*m.D*m.D + (sm.vocab ? 3*m.D*m.D + sm.vocab*m.D : 5*m.D*m.D + sm.layers*2*DRAFT_KVW*m.D) + (sm.rank ? 2*sm.rank*m.V*tpA : 0))*2/tpA : 0;
  const weights = wAttn+wLin+wExp+wMoeRep+wDense+wEmb+wMtp+wDraft;
  const nSwa = f.n>0 ? (f.swa||0) : 0;
  const kvTok = (f.n>0 ? (f.n-nSwa)*(r.kvTokRank + (r.idxTok||0)) : 0) + (sm ? sm.layers*2*DRAFT_KVW*2/Math.min(tpA,8) : 0);
  const swaPer = nSwa*r.kvTokRank*Math.min(c.n, f.W||0);          // 滑窗层每个请求最多存 W 个 token 的 KV
  const slots = c.engine==="sglang" ? 1 + (c.prefixCache?1:0) + (c.specK>0?4:0) : c.engine==="tokenspeed" ? 2 : 1 + (c.specK||0);
  const statePer = l.n>0 ? l.n*(r.Sl + r.Cl)*slots : 0;
  const total = hw.mem*1e9;
  const avail = total*c.memFrac - weights - c.reserveGB*1e9;
  const perSeq = statePer + swaPer + kvTok*c.n;
  const Bmax = Math.max(0, Math.floor(avail/Math.max(perSeq,1)));
  return { weights, wAttn, wLin, wExp, wMoeRep, wDense, wEmb, wMtp, wDraft, kvTok, swaPer, statePer, slots, perSeq, avail, total, Bmax };
}

return { ENGINES, SPEC, step, decode, prefill, memory, cover };
})();
