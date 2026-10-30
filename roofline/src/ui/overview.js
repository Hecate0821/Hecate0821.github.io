/* ================= 总览：推理一步和训练一步并排（照搬两边的当前配置） ================= */
const OGRP = { attn:"全注意力", lin:"线性注意力", moe:"MoE / FFN", lm:"lm_head / 采样", misc:"norm / 其他算子" };
function renderOverview(){
  const {m,hw,topo}=ctx(), ci=S.inf, ct=S.trn;
  syncInfRack(); syncTrnRack();      // 让预设和引擎相关的取值先归一（参数架本身在这一页不显示）
  const ri=infRun({phase:"decode"},"decode"), rt=trnRun(), e=m.moe;
  // 推理：按类别归并一次 decode
  const gi={}, gt={}; Object.keys(OGRP).forEach(k=>{ gi[k]=0; gt[k]=0; });
  for(const o of ri.ops) if(o.overlap!=="moe") gi[o.group]+=o.t;
  const tbo=sum(ri.comm.filter(x=>x.overlap==="tbo"),x=>x.t);
  const commI=sum(ri.comm.filter(x=>!x.overlap),x=>x.t)+Math.max(0,tbo-ri.tComp*0.5)+(ri.exposedOverlap||0);
  const specI=(ri.draftT||0)+(ri.commitT||0), otherI=Math.max(0,ri.stepT-sum(Object.values(gi))-commI-specI);
  // 训练：每个 PP stage 的一个 micro-batch
  for(const o of rt.ops) gt[o.group]+=(o.fwd.t+o.dgrad.t+o.wgrad.t)/rt.pp;
  const recT=rt.tR/rt.pp, commT=Math.max(0,rt.tMB-sum(Object.values(gt))-recT);
  const ratio=rt.tokPerGpu/ri.tokPerGpu;     // 生成一个 token 的卡秒 ÷ 训练一个 token 的卡秒
  $("o-lede").innerHTML="同一个 "+esc(m.short)+"，"+hw.name+" · "+esc(topo.name)+"。推理照搬推理页当前的 decode 配置，训练照搬后训练页当前的配置；改配置去对应的页，这里跟着变。";
  kpis($("o-kpis"),[
    ["推理 TPOT",fT(ri.tpot),ENGN[ci.engine]+"，每个 DP rank "+ri.Breq+" 个请求"],
    ["推理吞吐",fN(ri.tokPerGpu)+" tok/s/GPU","decode"],
    ["训练 step",fT(rt.stepT),(ct.fw==="megatron"?"Megatron-Core":"torchtitan")+"，"+fN(rt.tokens)+" token"],
    ["训练吞吐",fN(rt.tokPerGpu)+" tok/s/GPU","MFU "+(rt.mfu*100).toFixed(1)+"%"],
    ["生成 ÷ 训练",fX(ratio),"每个 token 的卡秒之比"]]);
  const lamI=e.E?ri.Tg*e.k/e.E:0, lamT=e.E?ct.tok*rt.ep*e.k/e.E:0;
  const memI=sum(ri.ops.filter(o=>o.bound==="memory"),o=>o.t)/ri.stepT;
  const passes=rt.ops.flatMap(o=>[o.fwd,o.dgrad,o.wgrad]), memT=sum(passes.filter(p=>p.bound==="memory"),p=>p.t)/Math.max(1e-30,sum(passes,p=>p.t));
  const tOf=o=>o.fwd.t+o.dgrad.t+o.wgrad.t, slowI=[...ri.ops].sort((a,b)=>b.t-a.t)[0], slowT=[...rt.ops].sort((a,b)=>tOf(b)-tOf(a))[0], nm=s=>esc(s.replace(/（.*?）/g,""));
  const usedI=ri.mem.weights+Math.min(ri.Breq,Math.max(0,ri.mem.Bmax))*ri.mem.perSeq+ci.reserveGB*1e9;
  table($("o-cmp"),["","推理：一步 decode","后训练：一个 step"],[
    ["框架与并行",esc(ENGN[ci.engine])+"，实例 "+ci.N+" 卡：attention TP "+(+ri.tpA.toFixed(2))+" × DP "+ri.dpA+"，MoE "+esc(MOENAME[ri.moePar]),(ct.fw==="megatron"?"Megatron-Core":"torchtitan")+"：TP "+rt.tp+" × CP "+rt.cp+" × PP "+rt.pp+" × DP "+(+rt.dp.toFixed(2))+"，EP "+rt.ep],
    ["精度","routed expert "+ci.wDt.toUpperCase()+"，其余 "+ci.dDt.toUpperCase()+"，KV "+ci.kvDt.toUpperCase(),esc((Train.PREC[ct.prec]||Train.PREC.bf16).name)],
    ["dense GEMM 的行数 M",fInt(ri.rowsG)+"（"+ri.Breq+" 个请求 × 宽度 "+ri.w+(ri.graph?"，graph 补齐后":"")+"）",fInt(ct.tok)+"（每卡每个 micro-batch）"],
    ...(e.E?[["每个 expert 平均几行",fI(lamI),fI(lamT)]]:[]),
    ["memory-bound 的时间",pct(memI,1),pct(memT,1)],
    ["最慢的算子",nm(slowI.name)+" "+fT(slowI.t),nm(slowT.name)+" "+fT(tOf(slowT))],
    ["没藏住的通信",pct(commI,ri.stepT)+" 的一步",pct(commT,rt.tMB)+" 的 micro-batch"],
    ["每卡显存",fB(usedI)+" / "+hw.mem+" GB",fB(rt.mem.used)+" / "+hw.mem+" GB"],
    ["吞吐",fN(ri.tokPerGpu)+" tok/s/GPU",fN(rt.tokPerGpu)+" tok/s/GPU"]]);
  stackBar($("o-ibar"),$("o-ibarlist"),[...Object.keys(OGRP).map(k=>({label:OGRP[k],v:gi[k],color:GROUPC[k]})),
    {label:"没藏住的通信",v:commI,color:css("c-a2a"),tag:pill("comm")},{label:"投机：draft + 提交",v:specI,color:css("c-cp")},
    {label:"launch / graph replay / DP 同步",v:otherI,color:css("c-free"),tag:pill("launch")}],ri.stepT,fT);
  stackBar($("o-tbar"),$("o-tbarlist"),[...Object.keys(OGRP).map(k=>({label:k==="lm"?"lm_head / CE":OGRP[k],v:gt[k],color:GROUPC[k]})),
    {label:"重算的前向",v:recT,color:css("c-cp")},{label:"没藏住的通信",v:commT,color:css("c-a2a"),tag:pill("comm")}],rt.tMB,fT);
  const rows=ri.rowsG>0?ct.tok/ri.rowsG:0;
  $("o-notes").innerHTML=ul([
    "<b>行数差 "+V(fI(rows)+"×")+"。</b>推理 decode 的 dense GEMM 只有 "+fInt(ri.rowsG)+" 行，训练每卡每个 micro-batch 有 "+fInt(ct.tok)+" 行"+(e.E?"；落到每个 expert 上是 "+fI(lamI)+" 对 "+fI(lamT)+" 行":"")+"。所以推理大半是 memory-bound（"+pct(memI,1)+"），训练大半是 compute-bound（memory-bound 只占 "+pct(memT,1)+"）。两边的量化收益因此完全不同：推理只量化权重就快（<a href=\"#infer/quant\">低精度推理</a>），训练要激活和梯度也降精度才快（<a href=\"#train/quant\">低精度训练</a>）。",
    "<b>并行方式不能共用。</b>推理切的是权重读取和 KV 容量（TP、DP-attention、EP、PD 分离），训练切的是激活、参数和长序列（TP + SP、CP、EP、FSDP / 分布式优化器、PP）。推理的见<a href=\"#infer/deploy\">部署与并行</a>，训练的见<a href=\"#train/par\">后训练 · 并行</a>。",
    "<b>RL 的账。</b>每个生成的 token 训练一遍时，生成一个 token 的卡秒是训练一个 token 的 "+V(fX(ratio))+(ratio>1?"，rollout 是大头，所以低精度 rollout 和投机解码对 RL 的收益最大（<a href=\"#rl/lowprec\">低精度 rollout</a>、<a href=\"#infer/spec\">投机解码</a>）":"，当前配置下训练更贵")+"；代价是两边算的不再是同一个函数（<a href=\"#rl/mismatch\">训推不一致</a>）。"]);
}
