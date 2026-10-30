/* ---------- 推理 · 部署与并行：推荐配置的取舍（同一个 Infer.decode / prefill） ---------- */
const TO = { seq:"8k1k", target:50, total:false };
const SEQS = { "1k1k":[1024,1024,"1K / 1K"], "1k8k":[1024,8192,"1K / 8K（推理链长）"], "8k1k":[8192,1024,"8K / 1K（InferenceX 常用）"], "32k2k":[32768,2048,"32K / 2K（agent 中段）"], "128k1k":[131072,1024,"128K / 1K（长上下文）"] };
/* DeepInfer 的形状：prefill 取放得下的最小 EP（从 2 起翻倍），decode 取拓扑里能拿到的最宽 EP（最多 64），attention 全 DP，PD 分离 */
function diConfig(mk, topoKey, hwKey){
  const m=mk===S.modelKey?S.model:MODELS[mk], hw=HW[hwKey||S.hw], G=TOPO[topoKey].gpus; if(!m.moe.E) return null;
  const w=Common.derive(m).total*Common.dtB(m.wDefault); let p=2; while(w/p>0.6*hw.mem*1e9 && p<64) p*=2;
  const N=Math.min(64,G), f8=m.wDefault==="fp8";
  return { eng:"DeepInfer", id:"deepinfer", label:"PD：prefill DEP"+p+" + decode DEP"+N, pd:{N:Math.min(p,G)},
    src:"DeepInfer 的部署建议套到这个模型：prefill 用放得下的最小 EP（这里 "+p+"），decode 用 EP64–128 里能拿到的最大值（这个拓扑最多 "+N+"），attention 全 DP，DeepEP，PD 分离，不用 TP 和 CP。DeepInfer 本身只服务 DeepSeek-V4.1-Flash，这里只借它的形状，按 SGLang 的 kernel 序列算。"+(N>TOPO[topoKey].domain?"这个拓扑上 decode 的 EP 要跨 NVLink 域走 IB。":""),
    cfg:{ engine:"sglang", N, dpA:N, moePar:"deepep", B:96, wDt:m.wDefault, dDt:f8?"fp8":"bf16", kvDt:"fp8", stateDt:"fp32" } };
}
function toConfigs(topoKey, hwKey){
  const L=[];
  for(const eng of ["sglang","vllm"]) for(const p of infPresets(eng,S.modelKey,topoKey)) if(p.cfg && p.cfg.N<=TOPO[topoKey].gpus)
    L.push({ eng:ENGN[eng], id:eng+":"+p.id, preset:p.id, engine:eng, label:p.label, src:p.src, cfg:Object.assign({engine:eng},p.cfg,{moePar:compatMoe(eng,p.cfg.moePar)}) });
  const di=diConfig(S.modelKey,topoKey,hwKey); if(di) L.push(di);
  return L;
}
const toShort = cf => cf.eng+" "+cf.label.split(/[，：（(]/)[0];
/* 一条 Pareto 曲线：每个 DP rank 的并发从 1 扫到显存 / 引擎上限 */
function toCurve(cf, isl, osl, hwKey, topoKey){
  const {m,g}=ctx(), hw=HW[hwKey||S.hw], topo=TOPO[topoKey||S.topo];
  const base=Object.assign(clone(BASE_INF),cf.cfg,{phase:"decode"});
  const probe=Infer.decode(m,hw,topo,Object.assign({},base,{B:1,n:isl+osl}),g), E=Infer.ENGINES[base.engine];
  const cap=Math.min(probe.mem.Bmax, base.specK>0?E.specCap:256), pts=[];
  if(probe.mem.avail<=0 || cap<1) return { pts, fit:false, wt:probe.mem.weights, cap:Math.max(0,cap), mem:probe.mem };
  const preC=Object.assign({},base,{phase:"prefill",specK:0,C:Math.min(isl,8192),P:isl,p:0});
  const pre=cf.pd ? Infer.prefill(m,hw,topo,Object.assign({},preC,{N:cf.pd.N,dpA:cf.pd.N}),g) : Infer.prefill(m,hw,topo,preC,g);
  const ladder=[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256].filter(b=>b<cap).concat([cap]);
  for(const b of ladder){
    const r=Infer.decode(m,hw,topo,Object.assign({},base,{B:b,n:isl+osl/2}),g), out=r.tokPerStep*r.dpA;
    /* 聚合部署：每步完成 out/osl 个请求，要顺带 prefill 同样多个新请求（摊到 DP rank 上）；PD 分离：prefill 卡按卡秒摊进分母 */
    const preT = cf.pd ? 0 : out/osl*pre.ttft/Math.max(1,r.dpA), t=r.stepT+preT;
    const gpus = cf.pd ? base.N + out/(t*osl)*pre.ttft : base.N;   // prefill 实例 DEP：每张卡一条 prompt，一条占 ttft 卡秒
    pts.push({ b, B:b*r.dpA, u:r.tokPerStep/r.Breq/t, thr:out/t/gpus*(TO.total?(isl+osl)/osl:1), t, preT, r });
  }
  return { pts, fit:true, wt:probe.mem.weights, cap, mem:probe.mem, pre };
}
function isoThr(pts, u){ let best=null;
  for(let i=0;i<pts.length;i++){ const p=pts[i], q=pts[i+1];
    if(p.u>=u) best=Math.max(best??0,p.thr);
    if(q && p.u>=u && q.u<u){ const a=(p.u-u)/(p.u-q.u); best=Math.max(best??0,p.thr+a*(q.thr-p.thr)); } }
  return best; }
function renderInferTO(){
  const {m,hw,topo,g}=ctx(), [isl,osl,seqName]=SEQS[TO.seq];
  fillOpts($("i-to-seq"),Object.entries(SEQS).map(([k,v])=>[k,v[2]]),TO.seq);
  fillOpts($("i-to-target"),[10,20,50,100,200,400].map(v=>[String(v),v+" tok/s/user"]),String(TO.target));
  $("i-to-total").checked=TO.total;
  const cfgs=toConfigs(S.topo), curves=cfgs.map((cf,i)=>Object.assign({cf,col:cf.eng==="DeepInfer"?"c-lin":"s"+Math.min(i+1,6)},toCurve(cf,isl,osl)));
  const ser=curves.filter(cu=>cu.fit).map(cu=>({name:cu.cf.eng+" · "+cu.cf.label,short:toShort(cu.cf),col:cu.col,pts:cu.pts.map(p=>[p.u,p.thr,{note:"每 rank 并发 "+p.b}])}));
  const yl=TO.total?"单卡吞吐（输入 + 输出）tok/s/GPU":"单卡吞吐（输出）tok/s/GPU";
  lineChart("i-to-pareto",{series:ser,xLab:"单用户速度 tok/s/user",yLab:yl,vlines:[{x:TO.target,label:"目标 "+TO.target}],nearest:true,empty:"没有放得下的配置"});
  legendInto("i-to-pareto-legend",ser.map(se=>({name:se.name,col:se.col,line:true})));
  const iso=cu=>cu.fit?isoThr(cu.pts,TO.target):null, cur=curves.find(cu=>cu.cf.engine===S.inf.engine&&cu.cf.preset===S.infPreset), best=curves.map(cu=>({cu,v:iso(cu)})).filter(z=>z.v!=null).sort((a,b)=>b.v-a.v)[0];
  const fast=curves.filter(cu=>cu.fit&&cu.pts.length).sort((a,b)=>b.pts[0].u-a.pts[0].u)[0];
  kpis($("i-to-kpis"),[
    ["@"+TO.target+" tok/s/user 最高",best?fN(best.v)+" tok/s/GPU":"达不到",best?toShort(best.cu.cf):"所有配置都更慢"],
    ["单用户最快",fast?fN(fast.pts[0].u)+" tok/s":"—",fast?toShort(fast.cf)+"，每 rank 1 个并发":""],
    ["单卡吞吐最高",(()=>{ const z=curves.filter(cu=>cu.fit).map(cu=>({cu,v:Math.max(...cu.pts.map(p=>p.thr))})).sort((a,b)=>b.v-a.v)[0]; return z?fN(z.v)+" tok/s/GPU":"—"; })(),"不限单用户速度"],
    ["推理页当前配置",cur?(iso(cur)!=null?fN(iso(cur))+" tok/s/GPU":"达不到"):"不在列表里",cur?toShort(cur.cf)+" @"+TO.target:"自定义或 "+ENGN[S.inf.engine]]]);
  $("i-to-cfg").innerHTML='<thead><tr><th></th><th>配置</th><th>卡 / 实例</th><th>attention</th><th>MoE</th><th>每卡权重</th><th>并发上限 / rank</th><th>@'+TO.target+'</th><th>来源</th><th></th></tr></thead><tbody>'+
    curves.map(cu=>{ const c=cu.cf.cfg, tp=c.N/c.dpA, on=cu===cur;
      return '<tr'+(on?' class="hl"':'')+'><td><span class="sw" style="display:inline-block;width:12px;height:3px;background:'+css(cu.col)+'"></span></td><td style="min-width:12em">'+esc(cu.cf.eng+" · "+cu.cf.label)+'</td><td class="num">'+c.N+(cu.cf.pd?" + "+cu.cf.pd.N+"（prefill）":"")+'</td><td>'+(tp>1?"TP"+tp:"")+(c.dpA>1?(tp>1?" × ":"")+"DP"+c.dpA:"")+(tp===1&&c.dpA===1?"单卡":"")+(c.dcp>1?" + DCP"+c.dcp:"")+(c.specK?"，投机 "+c.specK:"")+'</td><td>'+(m.moe.E?esc(MOENAME[c.moePar]):"—")+'</td><td class="num">'+fB(cu.wt)+'</td><td class="num">'+(cu.fit?cu.cap:"放不下")+'</td><td class="num">'+(iso(cu)!=null?fN(iso(cu)):"—")+'</td><td style="min-width:20em;font-size:12.5px">'+esc(cu.cf.src)+'</td><td>'+(on?"当前":'<button type="button" class="mini" data-to-apply="'+esc(cu.cf.id)+'">在推理页打开</button>')+'</td></tr>'; }).join("")+'</tbody>';
  /* 随上下文 */
  const isls=[1024,2048,4096,8192,16384,32768,65536,131072];
  const serC=curves.map(cu=>({name:cu.cf.eng+" · "+cu.cf.label,short:toShort(cu.cf),col:cu.col,pts:isls.map(L=>{ const c2=toCurve(cu.cf,L,osl); return [L,c2.fit?isoThr(c2.pts,TO.target):null]; })}));
  lineChart("i-to-ctx",{series:serC,xLog:true,xBase:2,xFmt:fK,xLab:"输入长度（输出 "+fK(osl)+"）",yLab:"@"+TO.target+" tok/s/user 的单卡吞吐",labelEnds:false,empty:"所有配置都达不到这个单用户速度"});
  legendInto("i-to-ctx-legend",serC.map(se=>({name:se.name,col:se.col,line:true})));
  $("i-to-ctx-lede").textContent="每个点是把这个输入长度下的 Pareto 曲线在 "+TO.target+" tok/s/user 处插值（InferenceX 的 iso-interactivity 读法）；断掉的地方是 KV 放不下，或者并发降到 1 也达不到这个速度。上下文变长时两件事同时发生：每个请求的 KV 读取变多（decode 每步变慢），能放下的并发变少（单卡吞吐的上限变低），聚合部署里分摊进来的 prefill 也变重。滑窗、线性注意力、MLA、DCP 压的是前两项，PD 分离把第三项挪到另一组卡上。";
  /* 怎么选 */
  const di=curves.find(cu=>cu.cf.eng==="DeepInfer"), b1=di&&di.fit?di.pts[0].r:null;
  const diNote = !di ? " dense 模型没有这条路。" : !b1 ? " 这个拓扑 / 卡上它放不下。" : (()=>{ const mm=b1.mem, nonExp=mm.wAttn+mm.wLin+mm.wMoeRep+mm.wDense+mm.wEmb, tW=nonExp/(hw.hbm*1e9*g.etaM);
    return " 它还有一个前提：attention 不切，每张卡每步都要读一遍全部非 expert 权重。"+esc(m.short)+" 的这部分每卡 "+fB(nonExp)+"，光读它就要 "+fT(tW)+"；并发 1 时一步 "+fT(b1.stepT)+"，单用户速度上限约 "+fN(di.pts[0].u)+" tok/s/user。"+(tW>0.01?"这个形状是按 attention 很小的 DeepSeek-V4.1-Flash 设计的，套到这里不合适：SGLang 的大规模预设在 attention 里保留了 TP8（attn-dp = N/8），就是为了把这部分切开。":"这部分足够小，宽 EP 的形状成立。"); })();
  $("i-to-rules").innerHTML=[
    "<b>先看放不放得下。</b>"+esc(m.short)+" 的权重是 "+fB(Common.derive(m).total*Common.dtB(m.wDefault))+"（"+m.wDefault.toUpperCase()+"），单卡 "+hw.mem+" GB。实例越小，剩给 KV 的越少、并发上限越低；上表“并发上限 / rank”就是这个数。",
    "<b>TP 买的是单用户速度。</b>decode 小并发时一步 ≈ 本卡权重字节 / HBM 带宽，TP 把它除以 N；代价是每层两次 all-reduce（小消息、受延迟限制），以及 KV 头数少于 TP 度时 KV 在每张卡各存一份。所以 TP 曲线在右侧（高 tok/s/user）占优，在左侧（高并发）被 DP 超过。",
    "<b>多实例 DP 买的是单卡吞吐。</b>独立副本之间不通信，单卡吞吐不随卡数下降；但每个副本都要读完整权重，单用户速度不会变快。模型放得进一张卡（Qwen3.5-35B、Qwen3.8-27B）时，TP1 × N 个实例就是 SGLang / vLLM 的推荐配置。",
    "<b>DP-attention + EP 买的是 KV 容量和 expert 的 batch。</b>attention 用 DP，KV 不复制；MoE 用 EP，每个 expert 收到所有 DP rank 的 token。代价是所有 rank 每步要对齐（空 rank 也要跑），每层两次 dispatch / combine。这里的 EP 通信和后训练用的是同一套链路模型（"+'<a href="#train/par/ep">后训练 · 并行 · EP</a>'+"）。",
    "<b>宽 EP + PD 分离（DeepInfer 的形状）买的是大规模下的单卡吞吐。</b>decode 用 64 卡以上的 EP，每卡只放几个 expert，省出的显存全给 KV；prefill 单独用小 EP。代价是最小部署单元很大、跨机时每层的 all-to-all 走 IB，而且 PD 分离本身不提高吞吐，只是让 decode 不受 prefill 干扰。"+diNote,
    best?"<b>当前设置（"+esc(seqName)+"，"+TO.target+" tok/s/user）下，</b>"+esc(best.cu.cf.eng+" · "+best.cu.cf.label)+" 的单卡吞吐最高："+fN(best.v)+" tok/s/GPU。换输入长度或目标速度，排序会变，这正是上面几张图要看的。":"<b>当前设置下没有配置能达到 "+TO.target+" tok/s/user。</b>"
  ].map(x=>"<li>"+x+"</li>").join("");
}
$("i-to-seq").addEventListener("change",e=>{ TO.seq=e.target.value; renderInferTO(); });
$("i-to-target").addEventListener("change",e=>{ TO.target=+e.target.value; renderInferTO(); });
$("i-to-total").addEventListener("change",e=>{ TO.total=e.target.checked; renderInferTO(); });
$("i-to-cfg").addEventListener("click",e=>{ const b=e.target.closest("[data-to-apply]"); if(!b) return;
  const cf=toConfigs(S.topo).find(x=>x.id===b.dataset.toApply); if(!cf) return;
  if(cf.preset){ S.inf.engine=cf.engine; applyInf(cf.preset); }
  else { S.inf=Object.assign(clone(BASE_INF),{n:S.inf.n,P:S.inf.P,C:S.inf.C},cf.cfg,{phase:"decode"}); S.infPreset="custom"; }
  syncInputs(); location.hash="infer/step"; render(); });

$("t-mesh-dim").addEventListener("click",e=>{ const b=e.target.closest("button[data-v]"); if(!b) return; meshDim=b.dataset.v; renderMesh(trnRun()); });
$("t-mesh").addEventListener("click",e=>{ const b=e.target.closest("button[data-k]"); if(!b) return; meshSel=+b.dataset.k; renderMesh(trnRun()); });
$("p-e-props").addEventListener("click",e=>{ const b=e.target.closest("[data-disp]"); if(!b) return; S.trn.dispatcher=b.dataset.disp; S.trnPreset="custom"; syncInputs(); render(); });
