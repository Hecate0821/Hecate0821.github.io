/* ================= 推理 · 投机解码：同一个 Infer.decode，只换 draft 方法、k 和 α ================= */
const SPEC_ORDER = ["mtp","eagle3","dflash","dspark"];
const SPECC = { mtp:"s1", eagle3:"s2", dflash:"s3", dspark:"s4" };
const SPEC_K = { mtp:3, eagle3:4, dflash:7, dspark:7 };          // 各方法常见的 draft 长度
const tauOf = (a,k) => a>=1 ? k+1 : (1-Math.pow(a,k+1))/(1-a);    // 每步平均产出（含 bonus），i.i.d. 接受
function alphaOf(tau,k){ if(k<=0||tau<=1) return 0; if(tau>=k+1) return 1;
  let lo=0, hi=1; for(let i=0;i<60;i++){ const mid=(lo+hi)/2; if(tauOf(mid,k)<tau) lo=mid; else hi=mid; } return (lo+hi)/2; }
/* 左侧没开投机时按 MTP、k = 3、α = 0.8 演示 */
function specNow(){ const c=S.inf, on=(c.specK||0)>0, k=on?c.specK:3;
  return { on, k, method:on?(c.specMethod||"mtp"):"mtp", a:on?alphaOf(1+c.accept,k):0.8 }; }
const specRun = (method,k,a,over) => infRun(Object.assign({phase:"decode",specMethod:method,specK:k,accept:k>0?tauOf(a,k)-1:0},over||{}),"decode");
const SPEC_ENG = [
  ["MTP",[["sglang","v0.5.11","Spec V2 overlap 调度成为默认"],["sglang","v0.5.13","V1 弃用，EAGLE / MTP 统一走 V2"]],[["vllm","v0.14.0","和 async scheduling 共用"],["vllm","v0.30.0","PP 下可用"]],"支持"],
  ["EAGLE-3",[["sglang","v0.5.5","DeepSeek 类模型"],["sglang","v0.5.7","SpecBundle 草稿权重"],["sglang","v0.5.12","SWA drafter、MLA"]],[["vllm","v0.12.0","多步 CUDA graph"],["vllm","v0.17.0","MRV2 + CUDA graph"]],"支持"],
  ["DFlash / DFlash2",[["sglang","v0.5.11","新增 DFlash"],["lmsys","2026-10-16-beam","DFlash2 + DP-attention（Beam）"]],[["vllm","v0.23.0","causal DFlash"],["vllm","v0.28.0","DFlash2"],["vllm","v0.31.0","DFlash + async scheduling"]],"—"],
  ["DSpark",[["sglang","v0.5.16","新增，按置信度定验证窗口"],["sglang","v0.5.20","PD 分离 + decode CP"]],[["vllm","v0.25.0","新增"],["vllm","v0.28.0","置信度调度验证、自适应预算"]],"—"],
  ["n-gram（不要 draft 模型）",[],[["vllm","v0.18.0","改到 GPU 上跑"],["vllm","v0.32.0","MRV2 的 GPU n-gram"]],"支持（NGram）"],
  ["独立 draft 模型",[],[["vllm","v0.15.0","新增"],["vllm","v0.31.0","MRV2 支持"]],"支持（draft / target）"],
  ["按负载开关投机",[["sglang","v0.5.12","自适应 Spec V2"]],[["vllm","v0.24.0","Dynamic SD"]],"—"]];
const TRTLLM_SPEC = "https://nvidia.github.io/TensorRT-LLM/features/speculative-decoding.html";

function renderSpec(){
  const {m,hw}=ctx(), c=S.inf, sn=specNow(), {k,a,method}=sn, P=Infer.ENGINES[c.engine], capped=isFinite(P.specCap), nm=Infer.SPEC[method].name;
  const r=specRun(method,k,a), base=specRun(method,0,a,{B:r.Breq}), tau=tauOf(a,k), verify=r.stepT-r.draftT-r.commitT;
  const warn=r.warn.filter(w=>!w.includes("max_running_requests"));      // 并发上限在正文里交代
  if(r.mem.avail<=0) warn.push("权重放不下：每卡权重 "+fB(r.mem.weights));
  else if(r.Breq>r.mem.Bmax) warn.push("B = "+r.Breq+" 超过开投机后显存能放下的 "+r.mem.Bmax+" 条（上下文 "+fInt(c.n)+"）");
  setWarn(warn);
  $("s-lede").innerHTML=(sn.on?"":"<b>左侧没开投机，这一页按 MTP、k = 3、α = 0.8 演示</b>（在左侧把 draft token 数调到 1 以上就跟着变）。")+
    esc(m.short)+" · "+hw.name+" · "+ENGN[c.engine]+"，每个 DP rank "+r.Breq+" 个并发"+(r.Breq<c.B?"（左侧设的是 "+c.B+"，"+ENGN[c.engine]+" 开投机后上限 "+P.specCap+"）":"")+"、上下文 "+fInt(c.n)+"。"+nm+" 每步 draft "+k+" 个 token，α = "+V(a.toFixed(2))+"，平均每步产出 τ = "+V(tau.toFixed(2))+" 个 token。"+
    "一步 = 验证（目标模型宽度 "+(k+1)+" 的 forward）+ draft + 提交；同样配置不开投机一步 "+fT(base.stepT)+"，产出 1 个 token。";
  const su=base.tpot/r.tpot, st=r.tokPerGpu/base.tokPerGpu;
  kpis($("s-kpis"),[
    ["TPOT",fT(r.tpot),"不开投机 "+fT(base.tpot)],
    ["单用户速度",fX(su),"相对不开投机",su<1],
    ["单卡吞吐",fN(r.tokPerGpu)+" tok/s/GPU",fX(st)+"，同样的并发",st<1],
    ["验证一步",fT(verify),"是不开投机一步的 "+fX(verify/base.stepT)],
    ["显存放得下",fInt(r.mem.Bmax)+" 条","不开投机 "+fInt(base.mem.Bmax)+(r.mem.wDraft>0?"；drafter "+fB(r.mem.wDraft)+" / 卡":r.mem.wMtp>0?"；MTP 层 "+fB(r.mem.wMtp)+" / 卡":""),r.mem.Bmax<r.Breq]]);
  stackBar($("s-bar"),$("s-barlist"),[
    {label:"验证：目标模型宽度 "+(k+1)+" 的 forward（含 lm_head、采样、通信、launch）",v:verify,color:css("c-attn")},
    {label:draftLabel({specMethod:method},k),v:r.draftT,color:css("c-cp")},
    {label:"接受后提交线性注意力状态",v:r.commitT,color:css("c-lin")}],r.stepT,fT);
  // τ 随 k
  const ks=[...Array(16).keys()], AS=[0.6,0.7,0.8,0.9];
  plot($("s-tau"),{h:320,x0:0,x1:15,y0:1,y1:Math.max(10,tau*1.1),fx:v=>String(v),fy:v=>String(+v.toPrecision(3)),xl:"draft 长度 k",yl:"每步平均产出 τ（token）",
    series:AS.map((x,i)=>({pts:ks.map(kk=>[kk,tauOf(x,kk)]),color:css("s"+(i+1)),w:Math.abs(x-a)<0.05?2.8:1.6,label:"α = "+x})),
    points:[{x:k,y:tau,color:css("ink"),r:6,label:"当前（α = "+a.toFixed(2)+"）"}]});
  $("s-tau-legend").innerHTML=AS.map((x,i)=>'<span style="color:'+css("s"+(i+1))+'"><i></i>α = '+x+'</span>').join("")+
    '<span>上限 τ = k + 1；α = 0.9 时 k 从 7 加到 15 只多产出 '+(tauOf(0.9,15)-tauOf(0.9,7)).toFixed(1)+' 个</span>';
  // 同一个 α，四种方法：左图扫 k，右图扫并发
  const one=[{name:"不开投机",short:"不开投机 = 1×",col:"muted"}];
  const serK=SPEC_ORDER.map(mt=>({name:Infer.SPEC[mt].name,short:Infer.SPEC[mt].name,col:SPECC[mt],pts:[1,2,3,4,5,6,7,8,10,12,15].map(kk=>[kk,base.tpot/specRun(mt,kk,a).tpot])}));
  lineChart("s-vsk",{series:serK.concat([Object.assign({pts:[[1,1],[15,1]]},one[0])]),xLab:"draft 长度 k",yLab:"单用户速度 ÷ 不开投机",xFmt:v=>String(v),yFmt:v=>v.toFixed(2)+"×",yMin:0,vlines:[{x:k,label:"当前 k"}]});
  const Bs=[1,2,4,8,16,24,32,48,64,96,128,192,256,384,512], b0={};
  const b0At=B=>b0[B]||(b0[B]=specRun(method,0,a,{B}));
  const serB=SPEC_ORDER.map(mt=>{ const kk=SPEC_K[mt], pts=[];
    for(const B of Bs){ const x=specRun(mt,kk,a,{B}); if(x.Breq<B||B>x.mem.Bmax) break; pts.push([B,x.tokPerGpu/b0At(B).tokPerGpu]); }
    return {name:Infer.SPEC[mt].name+"（k = "+kk+"）",short:Infer.SPEC[mt].name,col:SPECC[mt],pts}; });
  const xMax=Math.max(2,...serB.map(s=>s.pts.length?s.pts[s.pts.length-1][0]:1));
  lineChart("s-vsb",{series:serB.concat([Object.assign({pts:[[1,1],[xMax,1]]},one[0])]),xLog:true,xBase:2,xLab:"每个 DP rank 的并发 B",yLab:"单卡吞吐 ÷ 不开投机",xFmt:v=>fInt(v),yFmt:v=>v.toFixed(2)+"×",yMin:0,vlines:[{x:r.Breq,label:"当前 B"}]});
  // 右图的走势按当前方法那条线描述：dense 一路降；MoE 小并发时多激活 expert，先降后升
  const pts=serB[SPEC_ORDER.indexOf(method)].pts, at=p=>"B = "+p[0]+" 时 "+fX(p[1]), lo=pts[0], hi=pts[pts.length-1], best=pts.length?pts.reduce((b,p)=>p[1]>b[1]?p:b):null;
  let iMin=-1, iMax=-1;          // 第一个谷，和谷后的最高点（高出 5% 才算回升）
  for(let i=1;i<pts.length-1&&iMin<0;i++) if(pts[i][1]<=pts[i-1][1]&&pts[i][1]<=pts[i+1][1]){
    const j=pts.reduce((b,p,jj)=>jj>i&&p[1]>pts[b][1]?jj:b,i); if(pts[j][1]>pts[i][1]*1.05){ iMin=i; iMax=j; } }
  const dip=!!(m.moe&&m.moe.E) && iMin>0;
  const trend=!pts.length?"":dip
    ? "MoE 还多一层：k+1 个 token 各自路由，并发小时会多激活 expert、多读它们的权重，收益先降（"+nm+" 从 "+at(lo)+" 降到 "+at(pts[iMin])+"）；并发大到每层大部分 expert 本来就要读，这份代价就没了，收益回升（"+at(pts[iMax])+"）"+(iMax===pts.length-1?"，到右图截止处还在涨。":hi[1]<pts[iMax][1]*0.95?"，再往上 GEMM 进入 compute-bound，又回落（"+at(hi)+"）。":"，之后到右图截止处基本持平。")
    : "并发越大，验证宽度乘到每个 GEMM 的行数上，decode 从 memory-bound 走向 compute-bound，收益变小"+(pts.length>1&&hi[1]<lo[1]?"（"+nm+" 从 "+at(lo)+" 降到 "+at(hi)+"）":"")+"。";
  $("s-cost-lede").innerHTML="四种方法用同一个 α = "+V(a.toFixed(2))+"，只比代价：draft 读多少权重、算多少行。左图在当前并发 B = "+r.Breq+" 下扫 k，右图每种方法取常见的 k（MTP 3、EAGLE-3 4、DFlash / DSpark 7）扫并发。"+
    "真实的 α 各不相同：同一个目标模型上，DFlash / DSpark 的接受长度通常明显高于 EAGLE-3 和 MTP（见下面的公开数字），这里把它们拉平，只为看代价的形状。"+trend+
    (capped?ENGN[c.engine]+" 开投机后并发限到 "+P.specCap+"，右图到这里截止。":"右图扫到显存放不下为止。");
  legendInto("s-vs-legend",SPEC_ORDER.map(mt=>({name:Infer.SPEC[mt].name,col:SPECC[mt],line:true})).concat([{name:"不开投机 = 1×",col:"muted",line:true}]));
  table($("s-ledger"),["方法（α = "+a.toFixed(2)+"，B = "+r.Breq+"）","k","验证宽度","draft","验证","提交","一步","τ","TPOT","单用户速度","tok/s/GPU","draft 权重 / 卡","显存放得下的 B"],
    SPEC_ORDER.map(mt=>{ const kk=SPEC_K[mt], x=specRun(mt,kk,a);
      return {cls:mt===method?"hl":"",cells:[sw(css(SPECC[mt]))+Infer.SPEC[mt].name,kk,kk+1,fT(x.draftT),fT(x.stepT-x.draftT-x.commitT),x.commitT?fT(x.commitT):"—",fT(x.stepT),tauOf(a,kk).toFixed(2),fT(x.tpot),fX(base.tpot/x.tpot),fN(x.tokPerGpu),fB(x.mem.wDraft||x.mem.wMtp||0),fInt(x.mem.Bmax)]}; }),[1,2,3,4,5,6,7,8,9,10,11,12]);
  // 引擎支持、公开数字、RL
  const cell=list=>list.length?list.map(([p,v,t])=>relLink(p,v)+"："+esc(t)).join("<br>"):"—";
  table($("s-eng"),["方法","SGLang","vLLM",'TRT-LLM（PyTorch 后端，'+aOut(TRTLLM_SPEC,"文档")+'）'],SPEC_ENG.map(([n,sg,vl,tr])=>[esc(n),cell(sg),cell(vl),esc(tr)]));
  table($("s-pub"),["日期","来源","方法","模型 · 硬件","设置","数字"],DB.set("spec_public").slice().sort((x,y)=>y.date.localeCompare(x.date))
    .map(p=>['<span class="v">'+esc(p.date)+'</span>','<span class="nw">'+aOut(p.url,p.src)+'</span>',esc(p.method),esc(p.model)+(p.hw&&p.hw!=="—"?"<br>"+'<span class="tag">'+esc(p.hw)+'</span>':""),esc(p.setting),esc(p.value)]));
  // RL：长尾看右图左端；批量阶段和不开投机、跑满并发比
  const tailNote=!lo?"":"（右图左端："+nm+" 在 "+at(lo)+(lo===best?"，是整条线最高的地方":dip?"，MoE 在这里多读 expert，不如中等并发"+(lo[1]>1.02?"，但仍然净赚":""):"")+"）";
  const Bf=Math.min(c.B,Math.floor(base.mem.Bmax)), full=Bf>r.Breq?specRun(method,0,a,{B:Bf}):null, rf=full?r.tokPerGpu/full.tokPerGpu:0;
  const batch=full
    ? "批量阶段并发高："+ENGN[c.engine]+" 开投机后每个 DP rank 只能跑 "+r.Breq+" 个请求，不开投机能跑 "+Bf+" 个，单卡吞吐 "+fN(r.tokPerGpu)+" 对 "+fN(full.tokPerGpu)+" tok/s/GPU（"+fX(rf)+(rf<0.97?"，开了反而亏":rf<1.05?"，基本打平":"")+"）。"
    : "批量阶段并发高，验证把 GEMM 推进 compute-bound 后收益变小甚至为负。";
  $("s-rl").innerHTML=ul([
    "<b>长尾是投机最该开的地方。</b>一批 rollout 到最后只剩少数长轨迹，并发掉到个位数，整批的完成时间卡在这几条的单用户速度上"+tailNote+"；"+relLink("lmsys","2026-10-22-miles-mi355x-agentic-rl","LMSYS 的 SWE agent RL")+" 测到 32% 的 rollout 时间发生在 90% 的轨迹完成之后。"+batch+relLink("lmsys","2026-08-11-nemotron-3-5-lightning","Nemotron 3.5 Lightning")+" 的部署建议是追求最大吞吐时不开。所以要按负载开关："+relLink("sglang","v0.5.12")+" 的自适应 Spec V2、"+relLink("vllm","v0.24.0")+" 的 Dynamic SD。",
    "<b>draft 要跟着 policy 走。</b>策略每一步都在变，冻结的 MTP 层或 draft head 接受率会掉。"+relLink("lmsys","2025-11-19-miles","Miles")+" 在 RL 里在线 SFT draft 模型，rollout 比冻结的 MTP 快 25% 以上；"+relLink("slime","v0.2.0")+" 在 RL 里训练 MTP 层，"+relLink("miles","v0.1.3")+" 把训练后的 MTP 权重同步到 draft。推理侧要能热更新 draft："+relLink("sglang","v0.5.6")+"（MTP 层的 update_weights_from_tensor）、"+relLink("sglang","v0.5.21")+"（session 制权重更新连 draft 一起）、"+relLink("vllm","v0.26.0")+"（运行时更新 draft 权重）。",
    "<b>投机不改变采样分布，但改变 kernel 的形状。</b>拒绝采样保证输出分布和不开投机一样，不会多出一份 off-policy；但验证是宽度 k+1 的 forward，GEMM 的行数和 attention 的 query 数都和普通 decode 不同。要让 rollout 的 log-prob 和训练逐位一致（<a href=\"#rl/mismatch\">训推不一致</a>），验证也得走 batch-invariant 的 kernel：同一个 token 在宽度 1 和宽度 k+1 的 forward 里结果要一样。",
    "<b>显存。</b>外挂 drafter 有自己的权重和上下文 KV，SGLang 开投机还要给线性注意力多留状态槽。上表最后一列是开投机后显存放得下的并发，长上下文 rollout 时它可能比速度更先成为限制。"]);
}
