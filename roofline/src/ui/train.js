/* ================= 后训练 · 一个 step ================= */
const DISP = { megatron:[["alltoall","alltoall dispatcher（每 token 发 k 份）"],["deepep","flex + DeepEP（按 rank 去重）"],["hybridep","flex + HybridEP（去重，TMA / IBGDA）"],["ncclep","flex + NCCL EP（MXFP8 线上）"]],
  torchtitan:[["standard","NCCL all-to-all（k 份 + D2H 同步）"],["ag","AllGather + ReduceScatter（NCCL，静态形状）"],["deepep","DeepEP v2 expanded（按 rank 去重）"],["distmoe","Dist-MoE（对称显存，融合进 GEMM，SM100+）"],["mok","Mixture-of-Kittens（megakernel，GB NVL72）"]] };
const CPT = { megatron:[["p2p","p2p 环（默认，可藏在注意力后）"],["allgather","all-gather KV（--allgather-cp）"],["a2a","a2a（Ulysses）"]], torchtitan:[["allgather","all-gather KV（默认）"],["ring","Ring（P2P 环）"],["ulysses","Ulysses"]] };
const LCP = { megatron:[["headwise","按头 a2a（GDN 只能这样）"],["chunkwise","按段传 FP32 状态（KDA）"]], torchtitan:[["state","逐段传状态"],["ulysses","Ulysses"]] };
const RCO = { megatron:[["full","full（uniform，1 层一组）"],["selective","selective（按模块）"],["none","不重算"]], torchtitan:[["default","默认（只重算 w13）"],["full","full AC"],["none","不重算"]] };
const RCMODS = [["core_attn","core_attn"],["layernorm","layernorm"],["moe_act","moe_act"],["mla_up_proj","mla_up_proj"],["moe","moe（含 a2a 重放）"],["shared_experts","shared_experts"],["gdn","gdn"]];
function trnRun(over){ const {m,hw,topo,g}=ctx(); return Train.run(m,hw,topo,Object.assign({N:topo.gpus},S.trn,over||{}),g); }
function syncTrnRack(){
  const c=S.trn;
  qa("[data-fw]").forEach(el=>el.hidden=el.dataset.fw!==c.fw);
  for(const [id,L,k] of [["t-disp",DISP,"dispatcher"],["t-cptype",CPT,"cpType"],["t-lincp",LCP,"linCP"],["t-rc",RCO,"recompute"]]){
    if(!L[c.fw].some(x=>x[0]===c[k])) c[k]=L[c.fw][0][0]; fillOpts($(id),L[c.fw],c[k]); }
  $("t-rcmods").innerHTML=c.recompute==="selective"?'<p class="hint">--recompute-modules</p>'+RCMODS.map(x=>'<label class="check"><input type="checkbox" data-rcmod="'+x[0]+'"'+(c.rcModules.includes(x[0])?" checked":"")+'><span>'+x[1]+'</span></label>').join(""):"";
  presetSel($("t-preset"),$("t-preset-src"),trnPresets(c.fw,S.modelKey,S.topo),S.trnPreset);
}
function renderTrain(){
  const {m,d,hw,topo,g}=ctx(), c=S.trn, mg=c.fw==="megatron";
  syncTrnRack();
  const r=trnRun(), M=r.mem;
  const warn=[...r.warn]; if(M.used>M.total) warn.push("显存放不下："+fB(M.used)+" > "+hw.mem+" GB"); if(r.dp<1) warn.push("TP×CP×PP 超过了 "+topo.gpus+" 张卡");
  setWarn(warn);
  $("t-lede").innerHTML=esc(m.short)+" · "+hw.name+" · "+esc(topo.name)+" · "+(mg?"Megatron-Core":"torchtitan")+"。TP "+r.tp+(mg&&r.tp>1?"（SP）":"")+" × CP "+r.cp+" × PP "+r.pp+(r.vpp>1?"（VPP "+r.vpp+"）":"")+" × DP "+fI(r.dp)+"；EP "+r.ep+(mg?" × ETP "+r.etp:"")+" × expert-DP "+fI(r.edp)+"。每卡每 micro-batch "+fInt(c.tok)+" token，每个 step "+fInt(r.tokens)+" token、"+r.microsPerStep+" 个 micro-batch。";
  kpis($("t-kpis"),[
    ["step 时间",fT(r.stepT),"流水线 "+fT(r.tPipe)+" · 优化器 "+fT(r.tOpt)],
    ["吞吐",fN(r.tokPerGpu)+" tok/s/GPU","MFU "+(r.mfu*100).toFixed(1)+"%（按真实 FLOPs）"],
    ["一个 micro-batch",fT(r.tMB),"每 stage；计算 "+fT(r.compMB/r.pp)+"，通信露出 "+fT(r.exposed/r.pp)],
    ["PP 气泡",r.pp>1?pct(r.bubble,r.microsPerStep):"无",r.pp>1?"(p−1)/v = "+fI(r.bubble)+" 个 micro-batch":""],
    ["显存",fB(M.used),"上限 "+hw.mem+" GB",M.used>M.total],
    ["每卡参数",fN(r.PlD+r.PlE),"dense "+fN(r.PlD)+" · expert "+fN(r.PlE)]]);
  renderMesh(r); renderLayerGantt(r);
  // micro-batch 分解
  const pp=r.pp, ex=r.comm.filter(x=>!x.perStage), exSum=sum(ex,x=>x.exp||0), scale=exSum>0?r.exposed/exSum:0, byCls={};
  ex.forEach(x=>byCls[x.cls]=(byCls[x.cls]||0)+(x.exp||0)*scale);
  const clsName={tp:"TP 通信露出",cp:"CP 通信露出",ep:"EP a2a 露出",dp:"DP",pp:"PP p2p"};
  stackBar($("t-bar"),$("t-barlist"),[
    {label:"前向",v:r.tF/pp,color:css("c-attn")},{label:"反向 dgrad",v:r.tD/pp,color:css("c-moe")},{label:"反向 wgrad",v:r.tW/pp,color:css("c-lin")},{label:"重算的前向",v:r.tR/pp,color:css("c-cp")},
    ...Object.keys(byCls).map(k=>({label:clsName[k],v:byCls[k]/pp,color:CLSC[k],tag:pill("comm")})),
    {label:"FSDP2 参数 / 梯度通信露出",v:r.fsdpExposedMB,color:css("c-fsdp"),tag:pill("comm")},
    {label:"PP stage 间 p2p 露出",v:sum(r.comm.filter(x=>x.key==="pp"),x=>x.t)*0.1,color:css("c-dp"),tag:pill("comm")}],r.tMB,fT);
  $("t-step-lede").innerHTML="一个 step = "+r.microsPerStep+" 个 micro-batch 的流水线 + PP 气泡 + 没藏住的 DP 同步 + 优化器。"+esc(r.dpNote||"torchtitan FSDP2 的梯度同步已经在每个 micro-batch 里")+"；"+esc(r.optNote)+"。";
  stackBar($("t-sbar"),$("t-sbarlist"),[
    {label:"micro-batch × "+r.microsPerStep,v:r.tMB*r.microsPerStep,color:css("c-attn")},{label:"PP 气泡",v:r.tMB*r.bubble,color:css("c-free")},
    {label:"DP 梯度 reduce-scatter + 参数 all-gather 露出",v:r.tDPexp,color:css("c-fsdp"),tag:pill("comm")},{label:"优化器（"+r.optNote+"）",v:r.tOpt,color:css("c-moe")},
    {label:"标量 all-reduce（grad norm 等）",v:r.tSmallAR,color:css("c-ar"),tag:pill("comm")}],r.stepT,fT);
  // 逐算子
  table($("t-ops"),["算子（合计所在层，整模型）","层数","前向","dgrad","wgrad","重算","前向瓶颈","前向 I","三遍 FLOPs"],
    r.ops.map(o=>[sw(GROUPC[o.group])+esc(o.name),o.layers,fT(o.fwd.t),fT(o.dgrad.t),o.wgrad.t?fT(o.wgrad.t):"—",o.rc?"✓":"",pill(o.fwd.bound),fI(o.fwd.I),fF(o.fwd.fl+o.dgrad.fl+o.wgrad.fl)])
      .concat([{cls:"hl",cells:["<b>合计</b>","",fT(r.tF),fT(r.tD),fT(r.tW),fT(r.tR),"","",fF(r.flF+r.flB)]}]),[1,2,3,4,7,8]);
  // roofline
  const PR=Train.PREC[c.prec]||Train.PREC.bf16, pks=[{v:peakOf(hw,"bf16"),n:"BF16"}];
  for(const d of [PR.f,PR.b]) if(d!=="bf16" && !pks.some(p=>p.v===peakOf(hw,d))) pks.push({v:peakOf(hw,d),n:Common.peakKey(d).toUpperCase()});
  const pts=[], topT=new Set([...r.ops].sort((a,b)=>b.fwd.t-a.fwd.t).slice(0,8));
  for(const o of r.ops){ const col=GROUPC[o.group];
    if(o.fwd.fl>0) pts.push({x:o.fwd.I,y:o.fwd.fl/o.fwd.t,color:col,label:topT.has(o)?o.name.replace(/（.*?）/g,""):null});
    if(o.dgrad.fl>0) pts.push({x:o.dgrad.I,y:o.dgrad.fl/o.dgrad.t,color:col,shape:"sq",r:4});
    if(o.wgrad.fl>0) pts.push({x:o.wgrad.I,y:o.wgrad.fl/o.wgrad.t,color:col,shape:"tri",r:4}); }
  plot($("t-roof"),{h:380,x0:0.1,x1:20000,y0:1e10,y1:Math.max(...pks.map(p=>p.v))*3,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:"算术强度 I（FLOP / HBM 字节）",yl:"实际达到的 FLOP/s",
    series:roofSeries(hw,g,pks,0.1,20000),points:pts,vlines:pks.map((p,i)=>({x:p.v*g.etaC/(hw.hbm*1e9*g.etaM),color:css("muted"),label:p.n+" 拐点 "+fI(p.v*g.etaC/(hw.hbm*1e9*g.etaM)),dy:i*14}))});
  $("t-roof-legend").innerHTML='<span>● 前向　■ dgrad　▲ wgrad</span>'+Object.entries({attn:"全注意力",lin:"线性注意力",moe:"MoE / FFN",lm:"lm_head / CE",misc:"norm"}).map(([k,v])=>'<span><span class="sw" style="background:'+GROUPC[k]+'"></span>'+v+'</span>').join("");
  // 通信表
  const parOf=x=>x.par||({cp:"cp",ep:"ep"})[x.cls]||(x.key==="fsdp"?"fsdp":"");
  table($("t-comm"),["通信（每 micro-batch，每 rank）","维度","发出字节","时间","重叠方式","露出",""],
    r.comm.map(x=>[sw(CLSC[x.cls])+esc(x.name),x.cls.toUpperCase(),fB(x.bytes),fT(x.t),
      {moe:"藏在 expert GEMM 后",attn:"藏在注意力后",comp:"与计算重叠（按 85% 可藏）",prefetch:"预取，藏在计算后"}[x.ovl]||"串行",
      x.key==="fsdp"?fT(r.fsdpExposedMB):x.key==="pp"?fT(x.t*0.1):fT((x.exp||0)*scale), parOf(x)?'<a href="#train/par/'+parOf(x)+'">怎么算的</a>':""]),[2,3,5]);
  // 扫 token
  const toks=logspace(1024,65536,25).map(v=>Math.round(v/512)*512), ra=toks.map(t=>trnRun({tok:t}));
  plot($("t-sw-thr"),{x0:1024,x1:65536,y0:0,y1:Math.max(...ra.map(q=>q.tokPerGpu))*1.15,logx:true,fx:fxPow,fy:v=>fN(v),xl:"每卡每 micro-batch token",yl:"tok/s/GPU",
    series:[{pts:toks.map((t,i)=>[t,ra[i].tokPerGpu]),color:css("c-attn"),w:2.4}],vlines:[{x:c.tok,color:css("ink"),label:"当前"}],
    rects:toks.map((t,i)=>ra[i].mem.used>ra[i].mem.total?{x0:t*0.97,x1:t*1.15}:null).filter(Boolean)});
  plot($("t-sw-mem"),{x0:1024,x1:65536,y0:0,y1:Math.max(hw.mem*1.2,...ra.map(q=>q.mem.used/1e9*1.1)),logx:true,fx:fxPow,fy:v=>v+" GB",xl:"每卡每 micro-batch token",yl:"每卡显存",
    series:[{pts:toks.map((t,i)=>[t,ra[i].mem.used/1e9]),color:css("c-act"),w:2.4},{pts:toks.map((t,i)=>[t,ra[i].mem.act/1e9]),color:css("c-act"),w:1.2,dash:"5 4",label:"其中激活"},{pts:[[1024,hw.mem],[65536,hw.mem]],color:css("warn"),w:1.2,dash:"3 3",label:"HBM "+hw.mem+" GB"}],
    vlines:[{x:c.tok,color:css("ink"),label:"当前"}]});
  $("t-sw-legend").innerHTML='<span style="color:var(--c-attn)"><i></i>吞吐</span><span style="color:var(--c-act)"><i></i>显存（虚线：其中激活）</span><span>斜线阴影 = 显存放不下</span>';
  // 显存
  stackBar($("t-mem"),$("t-memlist"),[
    {label:mg?"本地参数 BF16 + FP32 main_grad":"—",v:M.paramsGPU,color:css("c-w")},{label:"optimizer 状态（分片）",v:M.states,color:css("c-e")},{label:"FSDP2 当前 + 预取块的全量参数与梯度",v:M.gathered,color:css("c-fsdp")},
    {label:"激活（"+(c.recompute==="full"?"full 重算：每层只存输入":"按保存的张量")+(r.pp>1?"，同时在飞 "+fI(M.inflight)+" 个 micro-batch":"")+"）",v:M.act,color:css("c-act")},
    {label:"logits"+(mg?"（FP32，不分块）":"（分块 CE）"),v:M.logits,color:css("c-lm")},{label:"EP a2a 缓冲",v:M.a2aBuf,color:css("c-a2a")},{label:"CP 收集的 KV",v:M.cpBuf,color:css("c-cp")},
    {label:"低精度权重缓存（"+esc(Train.PREC[c.prec].name)+"）",v:M.qW,color:css("c-moe")},
    {label:"workspace",v:M.work,color:css("c-free")},{label:"剩余",v:Math.max(0,M.total-M.used),color:css("soft")}],Math.max(M.total,M.used),fB);
  // 两个框架
  const cur=k=>c.fw===k?"（当前）":"";
  table($("t-diff"),["","Megatron-Core"+cur("megatron"),"torchtitan"+cur("torchtitan")],[
    ["参数怎么切","TP / PP 切参数；DP 组内只切 optimizer 状态（分布式优化器）","FSDP2 把参数、梯度、optimizer 状态都按 DP×CP 切，用时 all-gather"],
    ["每 micro-batch 的 DP 通信","没有；梯度在最后一个 micro-batch 的反向里 reduce-scatter，参数在下一个 step 前向时 all-gather","每个 block 都要 all-gather 参数（reshard 时反向再一次）+ reduce-scatter 梯度"],
    ["TP","TP + SP：每块前向 AG + RS，反向 3 AG + 1 RS；userbuffers 重叠","DTensor：Column AG + Row RS；async TP 选项；要求 EP ≥ TP"],
    ["CP","p2p 环 / all-gather / a2a / a2a+p2p；GDN 按头 a2a，KDA 可按段传状态","all-gather KV 或 Ulysses；线性注意力逐段传状态或 Ulysses"],
    ["EP","alltoall（k 份）或 flex（DeepEP / HybridEP / NCCL EP，按 rank 去重）；ETP 可 > 1","standard all-to-all（k 份 + host 同步）或 Dist-MoE（对称显存，和 GEMM 融合）"],
    ["重算","full uniform（slime / miles 全部配方）或按模块 selective；重算 moe 时 a2a 也重放","默认只重算 w13，不重放 a2a；也可 full AC"],
    ["输出头","vocab-parallel CE，FP32 logits 不分块","分块 CE"],
    ["线性注意力 kernel","FLA chunk_gated_delta_rule / chunk_kda","attn_gym（不依赖 fla）"],
    ["优化器","GPU Adam，或 CPU Adam（offload + D2H/H2D 重叠）","FSDP2 上的 GPU Adam，可 CPU offload"],
    ["MFU 怎么报","漏掉 KDA 和 MLA 层，GDN 少算","注意力不按 causal 打折"]]);
  // 解析
  const ops=[...r.ops].sort((a,b)=>(b.fwd.t+b.dgrad.t+b.wgrad.t)-(a.fwd.t+a.dgrad.t+a.wgrad.t));
  const A=[];
  A.push("一个 micro-batch 里最重的三个算子："+ops.slice(0,3).map(o=>esc(o.name.replace(/（.*?）/g,""))+" "+V(fT(o.fwd.t+o.dgrad.t+o.wgrad.t))).join("，")+"。反向（dgrad + wgrad）是前向的 "+V(fI((r.tD+r.tW)/Math.max(r.tF,1e-12))+"×")+(r.tR?"，重算再加 "+V(pct(r.tR,r.tF))+" 的前向":"")+"。");
  const commAll=sum(r.comm.filter(x=>!x.perStage),x=>x.t);
  A.push("通信合计 "+V(fT(commAll/pp))+"（每 stage），露出 "+V(fT(r.exposed/pp))+"。"+Object.keys(byCls).sort((a,b)=>byCls[b]-byCls[a]).slice(0,2).map(k=>clsName[k]+" "+V(fT(byCls[k]/pp))).join("，")+"。");
  if(r.ep>1) A.push("EP="+r.ep+"："+(["deepep","hybridep","ncclep","distmoe"].includes(c.dispatcher)?"按 rank 去重后每个 token 平均只发 "+V(fI(r.kEff))+" 份（top-k = "+m.moe.k+"）":"每个 token 发 k = "+m.moe.k+" 份")+"，每层 a2a "+V(fT(r.a2aPerLayer))+"，expert GEMM 每层 "+V(fT(r.moeT/d.nMoe))+"。"+(r.a2aPerLayer>r.moeT/d.nMoe?"a2a 比 expert 计算还长，换去重的 dispatcher 或打开 EP 重叠收益最大。":""));
  if(r.pp>1) A.push("PP="+r.pp+(r.vpp>1?"、VPP="+r.vpp:"")+"：气泡 "+V(pct(r.bubble,r.microsPerStep+r.bubble))+"。micro-batch 数 "+r.microsPerStep+" 由每步 token 决定，配方里 rollout batch 小时气泡会很明显。");
  A.push(mg?"Megatron 的 DP 同步只在 step 尾部做一次："+V(fT(r.tDP))+"，露出 "+V(fT(r.tDPexp))+"。":"torchtitan 的 FSDP2 每个 micro-batch 都要 all-gather 参数、reduce-scatter 梯度："+V(fT(r.fsdpMB))+"，露出 "+V(fT(r.fsdpExposedMB))+"。每卡 token 少时这部分藏不住。");
  A.push("优化器 "+V(fT(r.tOpt))+"（"+esc(r.optNote)+"），每卡分片参数 "+fN(r.Pshard)+"。"+(c.offload?"CPU Adam 省下 "+fB(r.Pshard*12)+" 显存，代价是 PCIe / C2C 往返。":""));
  if(M.used>M.total) A.push('<span style="color:var(--warn)">显存超了 '+fB(M.used-M.total)+'。最大的几项：'+[["激活",M.act],["参数",M.paramsGPU],["optimizer",M.states],["logits",M.logits]].sort((a,b)=>b[1]-a[1]).slice(0,2).map(x=>x[0]+" "+fB(x[1])).join("、")+'。</span>');
  if(m.lin.n>0) A.push(m.lin.kind+" 的 chunk kernel 前向 "+V(fT(sum(r.ops.filter(o=>o.key==="lcore"),o=>o.fwd.t)))+"、反向 "+V(fT(sum(r.ops.filter(o=>o.key==="lcore"),o=>o.dgrad.t)))+"：h 不保存，反向把整条递推重算一遍。");
  $("t-analysis").innerHTML=ul(A);
}
