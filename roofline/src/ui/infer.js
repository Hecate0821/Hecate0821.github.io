/* ================= 推理 · 一步 decode / prefill（参数架三节共用） ================= */
function infRun(over, phase){ const {m,hw,topo,g}=ctx(); const c=Object.assign({},S.inf,over||{}); return (phase||c.phase)==="decode" ? Infer.decode(m,hw,topo,c,g) : Infer.prefill(m,hw,topo,c,g); }
const infEngines = () => ["sglang","vllm","tokenspeed"].concat(S.inf.engine==="titan"?["titan"]:[]);
const asEng = e => ({engine:e, moePar:compatMoe(e,S.inf.moePar)});
/* 推理参数架：随引擎和阶段显示 / 隐藏，填 MoE 并行和预设（一步、部署、投机三节共用） */
function syncInfRack(){
  const c=S.inf;
  qa("[data-phase]").forEach(el=>el.hidden=el.dataset.phase!==c.phase);
  qa("[data-eng]").forEach(el=>el.hidden=el.dataset.eng!==c.engine);
  if(!MOEPAR[c.engine].some(x=>x[0]===c.moePar)) c.moePar=Infer.ENGINES[c.engine].defaultMoe;
  fillOpts($("i-moepar"),MOEPAR[c.engine],c.moePar);
  presetSel($("i-preset"),$("i-preset-src"),infPresets(c.engine,S.modelKey,S.topo),S.infPreset);
  $("i-engine-note").textContent={
    sglang:"一步要么全是 prefill，要么全是 decode；decode graph 包含 lm_head；开投机后并发限 48；Qwen / Kimi 融合 AR + RMSNorm。",
    vllm:"V1 默认把 prefill chunk 和 decode 放在同一步（这里分开显示）；graph 按 token 计；lm_head 和采样在 graph 外；开 DP 时每步一次 CPU 同步。",
    tokenspeed:"decode graph 只到 160（投机 80），超过就 eager；sampler 在 graph 里；默认 replicated-input EP；除 GLM 外都融合 AR。",
    titan:"vLLM 跑 torchtitan 的模型定义：没有 AR 融合、MLA 不 absorb、没有 prefill graph；batch-invariant 模式下训推 log-prob 逐位一致。"}[c.engine];
}
const specName = c => (Infer.SPEC[c.specMethod]||Infer.SPEC.mtp).name;
const draftLabel = (c,k) => c.specMethod==="dflash"||c.specMethod==="dspark" ? specName(c)+" draft（一次 forward 出 "+k+" 个，含上下文 K / V 投影和 lm_head）"
  : specName(c)+" draft "+k+" 步（"+(c.specMethod==="eagle3"?"含每步裁剪词表的 lm_head":"含每步 lm_head")+"）";
function renderInfer(){
  const {m,d,hw,topo,g}=ctx(), c=S.inf, dec=c.phase==="decode", E=Infer.ENGINES[c.engine];
  syncInfRack();
  const r=infRun(); const inst=Math.floor(topo.gpus/c.N);
  const warn=[...r.warn];
  if(topo.gpus%c.N) warn.push("实例 GPU 数 "+c.N+" 不能整除拓扑的 "+topo.gpus+" 张卡");
  if(c.N%c.dpA) warn.push("attention DP 必须整除 N");
  if(c.N>topo.gpus) warn.push("实例比拓扑还大");
  if(c.moePar==="fused" && c.engine==="titan" && !blackwell()) warn.push("Dist-MoE 需要 SM100+");
  if(r.mem.avail<=0) warn.push("权重放不下：每卡权重 "+fB(r.mem.weights)+"，显存 "+hw.mem+" GB × mem_fraction "+c.memFrac);
  else if(dec && r.Breq>r.mem.Bmax) warn.push("B="+r.Breq+" 超过显存能放下的 "+r.mem.Bmax+" 条（上下文 "+fInt(c.n)+"）");
  if(dec && !r.graph && c.graph) warn.push(ENGN[c.engine]+" 的 decode graph 只到 "+E.maxGraph(hw,r.specK>0)+(c.engine==="vllm"||c.engine==="titan"?" 个 token":" 个请求")+"，这一步走 eager");
  setWarn(warn);
  $("i-lede").innerHTML=esc(m.short)+" · "+hw.name+" · "+esc(topo.name)+" · "+ENGN[c.engine]+"。一个实例 "+c.N+" 卡（attention TP "+fI(r.tpA)+" × DP "+r.dpA+"，MoE "+MOENAME[r.moePar]+"），拓扑里放 "+inst+" 个实例。"+
    (dec? "每个 DP rank "+r.Breq+" 个并发请求，上下文 "+fInt(c.n)+" token"+(r.specK?"，"+specName(c)+" 投机 "+r.specK+" 个 draft、平均多接受 "+c.accept:"")+"。" : "一个 "+fInt(c.P)+" token 的 prompt，每 chunk "+fInt(c.C)+" token。");
  const wPer=r.mem.weights;
  if(dec) kpis($("i-kpis"),[
    ["TPOT",fT(r.tpot),"一步 "+fT(r.stepT)+(r.specK?" ÷ "+(1+c.accept).toFixed(1)+" token":"")],
    ["吞吐",fN(r.tokPerGpu)+" tok/s/GPU","整个拓扑 "+fN(r.tokPerGpu*topo.gpus)+" tok/s"],
    ["CUDA graph",r.graph?"补到 "+r.rowsG+" 行":"eager",r.graph?"真实 "+r.rowsR+" 行，多算 "+pct(r.rowsG-r.rowsR,r.rowsR):r.nKern+" 个 kernel × "+g.launchUs+" µs launch",!r.graph&&c.graph],
    ["显存放得下",fInt(r.mem.Bmax)+" 条","每 DP rank，上下文 "+fInt(c.n),r.mem.Bmax<r.Breq],
    ["每卡权重",fB(wPer),"expert "+fB(r.mem.wExp),r.mem.avail<=0],
    ["每请求缓存",fB(r.mem.perSeq),"KV "+fB(r.mem.kvTok*c.n)+(r.mem.statePer>0?" + 状态 "+fB(r.mem.statePer)+"（"+r.mem.slots+" 槽）":"")]]);
  else kpis($("i-kpis"),[
    ["TTFT",fT(r.ttft),r.chunks.length+" 个 chunk"],
    ["第一个 chunk",fT(r.chunks[0]?r.chunks[0].t:r.stepT),"最后一个 "+fT(r.chunks.length?r.chunks[r.chunks.length-1].t:0)],
    ["prefill 吞吐",fN(r.tokPerGpu)+" tok/s/GPU","按第一个 chunk"],
    ["prefill graph",r.graph?"有":"eager",E.prefillGraph(m)?"":"这个引擎对这个模型不开 prefill graph"],
    ["每卡权重",fB(wPer),"",r.mem.avail<=0]]);
  // 时间分解
  const items=[];
  for(const o of r.ops) if(o.overlap!=="moe") items.push({label:o.name,v:o.t,color:GROUPC[o.group],tag:pill(o.bound)});
  for(const x of r.comm) if(!x.overlap) items.push({label:x.name,v:x.t,color:COMMC[x.key]||css("net"),tag:pill("comm")});
  const tboC=sum(r.comm.filter(x=>x.overlap==="tbo"),x=>x.t);
  if(tboC>0) items.push({label:"TBO 后露出的 a2a",v:Math.max(0,tboC-r.tComp*0.5),color:css("c-a2a"),tag:pill("comm")});
  if(r.exposedOverlap>0) items.push({label:"shared expert / 融合 a2a 没藏住的部分",v:r.exposedOverlap,color:css("c-a2a"),tag:pill("comm")});
  const base=r.stepT-(r.draftT||0)-(r.commitT||0), dpSync=(c.engine==="vllm"||c.engine==="titan")&&r.dpA>1?g.dpSyncUs*1e-6:0;
  if(r.graph) items.push({label:"graph replay",v:g.graphLaunchUs*1e-6,color:css("c-free"),tag:pill("launch")});
  const cpuExtra=base-dpSync-r.gpu-(r.graph?g.graphLaunchUs*1e-6:0);
  if(cpuExtra>1e-9) items.push({label:"CPU 发 kernel 比 GPU 跑得慢的部分（eager，"+r.nKern+" 个 kernel）",v:cpuExtra,color:css("c-free"),tag:pill("launch")});
  if(dpSync) items.push({label:"DP 每步 CPU 同步",v:dpSync,color:css("c-free"),tag:pill("launch")});
  if(r.draftT) items.push({label:draftLabel(c,r.specK),v:r.draftT,color:css("c-cp")});
  if(r.commitT) items.push({label:"接受后提交线性状态",v:r.commitT,color:css("c-lin")});
  stackBar($("i-bar"),$("i-barlist"),items,r.stepT,fT);
  // kernel 表
  $("i-ops-lede").innerHTML="按执行顺序列出 "+ENGN[c.engine]+" 在这一步发出的 kernel（合计所有层）。"+(dec?"“按谁算”一列说明行数：权重 GEMM 和 norm 按 graph 补齐后的 "+r.rowsG+" 行，读 KV / 状态的 kernel 只按真实的 "+r.Breq+" 个请求，router 在 replicated-input EP 下按全局 "+fInt(r.Tg)+" 行。":"");
  const rowsOf=o=> !dec ? "—" : o.scalesWith==="real" ? r.Breq+" 请求" : (o.key==="router"&&r.moePar==="replicated") ? fInt(r.Tg)+" 全局" : o.key==="moe" ? "expert 行" : (o.key==="lm"||o.key==="samp") ? (E.lmInGraph&&r.graph&&o.key==="lm"||E.samplerInGraph&&r.graph&&o.key==="samp" ? r.rowsG+" 补齐" : r.rowsR+" 真实") : r.rowsG+" 补齐";
  const note=o=>[o.overlap==="moe"?"另一个 stream，藏在 routed expert 后":"", o.stateMult?"状态读写 × "+o.stateMult:"", o.touched!=null?"每卡碰到 "+fI(o.touched)+" 个 expert（覆盖 "+pct(o.cover,1)+"）":""].filter(Boolean).join("；");
  table($("i-ops"),["kernel（合计 L 层）","层数","按谁算","FLOPs","HBM 字节","I","瓶颈","时间","说明"],
    r.ops.map(o=>[sw(GROUPC[o.group])+esc(o.name),o.layers,rowsOf(o),fF(o.flops),fB(o.bytes),fI(o.I),pill(o.bound),fT(o.t),esc(note(o))]),[1,3,4,5,7]);
  table($("i-comm"),["通信","次数","每卡发出","时间","和谁重叠"],
    r.comm.length? r.comm.map(x=>[sw(COMMC[x.key]||css("net"))+esc(x.name),x.layers,fB(x.bytes),fT(x.t),x.overlap==="moe"?"融合进 expert GEMM":x.overlap==="tbo"?"two-batch overlap":"串行"]) : [["单卡实例，没有通信","","","",""]],[1,2,3]);
  // roofline
  const pks=[{v:peakOf(hw,"bf16"),n:"BF16"}]; if(dtB(c.wDt)<2) pks.push({v:peakOf(hw,c.wDt),n:c.wDt.toUpperCase()}); else if(dtB(c.dDt)<2) pks.push({v:peakOf(hw,c.dDt),n:"FP8"});
  const top=new Set([...r.ops].sort((a,b)=>b.t-a.t).slice(0,10));
  plot($("i-roof"),{h:380,x0:0.05,x1:20000,y0:1e9,y1:Math.max(...pks.map(p=>p.v))*3,logx:true,logy:true,fx:fxPow,fy:fyFlops,xl:"算术强度 I（FLOP / HBM 字节）",yl:"实际达到的 FLOP/s",
    series:roofSeries(hw,g,pks,0.05,20000),points:r.ops.filter(o=>o.flops>0).map(o=>({x:o.I,y:o.flops/o.t,color:GROUPC[o.group],label:top.has(o)?o.name.replace(/（.*?）/g,""):null})),
    vlines:pks.map((p,i)=>({x:p.v*g.etaC/(hw.hbm*1e9*g.etaM),color:css("muted"),label:p.n+" 拐点 "+fI(p.v*g.etaC/(hw.hbm*1e9*g.etaM)),dy:i*14}))});
  $("i-roof-legend").innerHTML='<span><i></i>BF16 屋顶（η 折算后）</span>'+(pks[1]?'<span><i class="d"></i>'+pks[1].n+' 屋顶</span>':'')+
    Object.entries({attn:"全注意力",lin:"线性注意力",moe:"MoE / FFN",lm:"lm_head / 采样",misc:"norm / 残差"}).map(([k,v])=>'<span><span class="sw" style="background:'+GROUPC[k]+'"></span>'+v+'</span>').join("")+'<span>贴着 0.1~1 µs 横线的点是被“最短 kernel 时间”卡住的</span>';
  // 换引擎
  const engs=infEngines(), runs={}; engs.forEach(e=>runs[e]=infRun(asEng(e)));
  if(dec){
    const bmx=Math.max(...engs.map(e=>runs[e].mem.Bmax)), x1=Math.max(16,Math.min(1024,Math.ceil(Math.max(bmx,r.Breq)*1.15))), st=Math.max(1,Math.round(x1/360));
    const lat=[], thr=[]; let ymaxL=0, ymaxT=0;
    for(const e of engs){ const pl=[], pt=[]; const bm=runs[e].mem.Bmax;
      for(let B=1;B<=x1;B+=st){ const q=infRun(Object.assign(asEng(e),{B}),"decode"); if(q.Breq<B) break; const ok=B<=bm;
        pl.push([B,ok?q.tpot*1e3:NaN]); pt.push([B,ok?q.tokPerGpu:NaN]); if(ok){ ymaxL=Math.max(ymaxL,q.tpot*1e3); ymaxT=Math.max(ymaxT,q.tokPerGpu);} }
      lat.push({pts:pl,color:ENGC[e],w:e===c.engine?2.8:1.6}); thr.push({pts:pt,color:ENGC[e],w:e===c.engine?2.8:1.6}); }
    const vl=[{x:r.Breq,color:css("ink"),label:"当前 B"}];
    plot($("i-sw-lat"),{x0:0,x1,y0:0,y1:Math.max(ymaxL,1e-3)*1.1,fx:v=>fInt(v),fy:v=>+v.toPrecision(3)+"",xl:"每个 DP rank 的并发请求 B",yl:"TPOT（ms）",series:lat,vlines:vl});
    plot($("i-sw-thr"),{x0:0,x1,y0:0,y1:Math.max(ymaxT,1)*1.1,fx:v=>fInt(v),fy:v=>fN(v),xl:"每个 DP rank 的并发请求 B",yl:"tok/s/GPU",series:thr,vlines:vl});
    $("i-sw-legend").innerHTML=engs.map(e=>'<span style="color:'+ENGC[e]+'"><i></i>'+ENGN[e]+'（MoE：'+MOENAME[compatMoe(e,c.moePar)]+'，显存上限 '+runs[e].mem.Bmax+'）</span>').join("")+'<span>曲线在各自的显存上限处截断</span>';
  } else {
    let ymax=0; const series=engs.map(e=>{ const pts=runs[e].chunks.map(ch=>[ch.p+ch.C/2,ch.t*1e3]); pts.forEach(p=>ymax=Math.max(ymax,p[1])); return {pts,color:ENGC[e],w:e===c.engine?2.8:1.6}; });
    plot($("i-chunks"),{x0:0,x1:c.P,y0:0,y1:Math.max(ymax,1e-3)*1.1,fx:v=>fxPow(v),fy:v=>+v.toPrecision(3)+"",xl:"chunk 在 prompt 里的位置（token）",yl:"chunk 耗时（ms）",series});
    $("i-chunks-legend").innerHTML=engs.map(e=>'<span style="color:'+ENGC[e]+'"><i></i>'+ENGN[e]+'：TTFT '+fT(runs[e].ttft)+'</span>').join("");
  }
  // 引擎差别表
  const yes=b=>b?"是":"否", P=e=>Infer.ENGINES[e];
  const R=[["一步里混 prefill 和 decode",e=>yes(P(e).mixed)],
    ["decode graph 上限",e=>P(e).maxGraph(hw,(c.specK||0)>0)+(e==="vllm"||e==="titan"?" token":" 请求")],
    dec?["这一步：graph / 补齐",e=>runs[e].graph?runs[e].rowsR+" → "+runs[e].rowsG+" 行":"eager"]:["prefill graph（这个模型）",e=>yes(runs[e].graph)],
    ["lm_head / 采样在 graph 里",e=>yes(P(e).lmInGraph)+" / "+yes(P(e).samplerInGraph)],
    ["层边界 AR + RMSNorm 融合（当前）",e=>runs[e].tpA>1?yes(runs[e].fused):"TP=1，无 AR"],
    ["默认 MoE 并行 / 这里用的",e=>MOENAME[P(e).defaultMoe]+" / "+MOENAME[runs[e].moePar]],
    ["开投机时的并发上限",e=>isFinite(P(e).specCap)?P(e).specCap:"不限"],
    ...(m.lin.n>0?[["线性状态流量（每层每请求）",e=>dec&&runs[e].specK>0?P(e).stateVerify(m,runs[e].w)+"·S_l":P(e).stateDecode(m)+"·S_l"],
      ["prefill chunk kernel 写回 h",e=>yes(P(e).hMat(m,hw))]]:[]),
    ...(m.full.kind!=="gqa"?[["MLA prefill 每 chunk 重展开前缀",e=>m.full.kind==="dsa"&&e==="tokenspeed"?"否（absorbed 稀疏）":yes(P(e).mlaReexpand)]]:[]),
    ["shared expert 重叠",e=>({graph:"graph 下另一个 stream",small:"T ≤ 256 时 aux stream",none:"无"})[P(e).sharedOverlap]],
    ["kernel 数",e=>fInt(runs[e].nKern)],
    ...(dec?[["单步",e=>fT(runs[e].stepT)],["TPOT",e=>fT(runs[e].tpot)],["tok/s/GPU",e=>fN(runs[e].tokPerGpu)],["显存放得下的 B",e=>fInt(runs[e].mem.Bmax)]]
      :[["TTFT",e=>fT(runs[e].ttft)],["prefill tok/s/GPU",e=>fN(runs[e].tokPerGpu)]])];
  table($("i-diff"),[""].concat(engs.map(e=>ENGN[e]+(e===c.engine?"（当前）":""))),R.map(row=>[row[0]].concat(engs.map(e=>row[1](e)))));
  // 显存
  const M=r.mem, tot=M.total, Bm = dec ? r.Breq*r.dpA/r.dpA : 1;
  const kvU=Math.min(Bm,Math.max(0,M.Bmax))*M.kvTok*c.n, stU=Math.min(Bm,Math.max(0,M.Bmax))*M.statePer;
  stackBar($("i-mem"),$("i-memlist"),[
    {label:"全注意力权重",v:M.wAttn,color:css("c-attn")},{label:"线性注意力权重",v:M.wLin,color:css("c-lin")},{label:"routed expert 权重",v:M.wExp,color:css("c-moe")},
    {label:"shared / router / latent 投影",v:M.wMoeRep,color:css("c-e")},{label:"dense FFN",v:M.wDense,color:css("c-e")},{label:"embedding + lm_head",v:M.wEmb,color:css("c-lm")},{label:"MTP 层",v:M.wMtp,color:css("c-cp")},{label:specName(c)+" drafter",v:M.wDraft||0,color:css("c-cp")},
    {label:"KV cache（"+Bm+" 请求 × "+fInt(c.n)+" token）",v:kvU,color:css("c-kv")},{label:"线性状态（"+Bm+" 请求 × "+M.slots+" 槽）",v:stU,color:css("c-st")},
    {label:"激活与 workspace 预留",v:c.reserveGB*1e9,color:css("c-act")},{label:"KV 池里还空着",v:Math.max(0,M.avail-kvU-stU),color:css("soft")},{label:"mem_fraction 之外",v:tot*(1-c.memFrac),color:css("c-free")}],tot,fB);
  // 解析
  const ops=[...r.ops].sort((a,b)=>b.t-a.t), memT=sum(r.ops.filter(o=>o.bound==="memory"),o=>o.t), launchT=sum(r.ops.filter(o=>o.bound==="launch"),o=>o.t), commT=sum(r.comm.filter(x=>!x.overlap),x=>x.t);
  const moeO=r.ops.find(o=>o.key==="moe"), kvO=r.ops.filter(o=>o.scalesWith==="real"), kvBytes=sum(kvO,o=>o.bytes);
  const A=[];
  A.push("最慢的三个：" + ops.slice(0,3).map(o=>esc(o.name.replace(/（.*?）/g,""))+" "+V(fT(o.t))+" "+pill(o.bound)).join("，")+"。");
  A.push("memory-bound 的 kernel 占 "+V(pct(memT,r.stepT))+"，被最短 kernel 时间卡住的占 "+V(pct(launchT,r.stepT))+"，没藏住的通信占 "+V(pct(commT,r.stepT))+"。");
  if(moeO&&dec) A.push("一步里全局 "+fInt(r.Tg)+" 行进 MoE，每卡碰到 "+V(fI(moeO.touched))+" 个 expert（覆盖率 "+pct(moeO.cover,1)+"），每层读 expert 权重 "+V(fB(moeO.bytes/moeO.layers))+"。B 再大，覆盖率也到头了，之后 expert GEMM 才开始受益于更大的 batch。");
  if(dec&&kvO.length) A.push("读 KV 和读写线性状态（只随真实请求数和上下文增长）一共 "+V(fB(kvBytes))+"，用时 "+V(fT(sum(kvO,o=>o.t)))+"。"+(m.lin.n>0&&r.specK>0?"开投机后线性状态的读写是普通 decode 的 "+(E.stateVerify(m,r.w)/E.stateDecode(m)).toFixed(1)+" 倍（"+ENGN[c.engine]+" 的验证方式）。":""));
  if(dec&&r.graph&&r.rowsG>r.rowsR) A.push("graph 把 "+r.rowsR+" 行补到 "+r.rowsG+" 行，补出来的行在权重 GEMM 里白算；decode 的权重 GEMM 是 memory-bound，所以代价不大，主要多在 lm_head 和激活读写。");
  if(dec&&!r.graph) A.push("这一步走 eager，"+r.nKern+" 个 kernel 每个 "+g.launchUs+" µs 的 CPU launch，合计 "+V(fT(r.nKern*g.launchUs*1e-6))+(r.cpuLaunch>r.gpu?"，比 GPU 时间还长，step 被 CPU 卡住。":"。"));
  if(r.tpA>1) A.push("attention TP="+fI(r.tpA)+"，层边界的 all-reduce "+(r.fused?"和残差 + RMSNorm 融合成一个 kernel":"没有融合，norm 单独一个 kernel")+"。"+(m.full.kind!=="gqa"?"MLA latent KV 每卡复制一份"+(c.dcp>1?"，DCP="+c.dcp+" 把它切成 "+c.dcp+" 份":"，TP 再大也不省 KV")+"。":m.full.Hkv<r.tpA?"KV 头只有 "+m.full.Hkv+" 个，TP="+fI(r.tpA)+" 时每卡复制。":""));
  if(c.engine==="tokenspeed" && dec && r.Breq>E.maxGraph(hw,r.specK>0)) A.push("TokenSpeed 的 graph 只到 "+E.maxGraph(hw,r.specK>0)+"，再加并发就是 eager，曲线在这里跳一下。");
  if(c.engine==="sglang" && r.specK>0) A.push("SGLang 开投机后 max_running_requests 被限到 48，大并发吞吐上不去；它换来的是单请求延迟。");
  if(c.engine==="vllm") A.push("vLLM 默认混批：真实服务里 decode 这一步常常带着一个 prefill chunk，单步时间会是两者之和。"+(r.dpA>1?"开 DP 后每步多一次 CPU 侧同步。":""));
  if(r.moePar==="replicated"&&r.N>1) A.push("TokenSpeed 的 replicated-input EP：每卡对全部 "+fInt(r.Tg)+" 行做 router，只算本地 expert，结果并进下一层的 all-reduce，不需要 a2a。");
  $("i-analysis").innerHTML=ul(A);
}
