/* ---------- 后训练核心的 Train.run 换参数 ---------- */
/* 后训练的并行一节按 torchtitan 的 FSDP + CP + EP 摆：框架是 Megatron 时，同样的 TP / CP / EP 度搬过去，PP 去掉、分布式优化器换成 FSDP2 */
const MG2TT = { disp:{alltoall:"standard",deepep:"deepep",hybridep:"deepep",ncclep:"standard"}, cp:{p2p:"ring",allgather:"allgather",a2a:"ulysses"} };
function parBase(){
  const c=S.trn; if(c.fw==="torchtitan") return clone(c);
  return Object.assign(clone(c), { fw:"torchtitan", pp:1, vpp:1, etp:1, dispatcher:MG2TT.disp[c.dispatcher]||"standard", cpType:MG2TT.cp[c.cpType]||"allgather",
    linCP:"state", recompute:c.recompute==="none"?"none":"full", offload:false, tp:Math.min(c.tp, Math.max(1,c.ep)) });
}
function runP(over, topoKey, hwKey){ const {m,g}=ctx(), topo=TOPO[topoKey||S.topo], hw=HW[hwKey||S.hw];
  return Train.run(m, hw, topo, Object.assign({N:topo.gpus}, parBase(), over||{}), g); }
const BADCFG = /必须|要求|只支持|只能|面向/;
const runOK = r => r.dp>=1 && Number.isInteger(r.dp) && !r.warn.some(w=>BADCFG.test(w));
const runFit = r => r.mem.used <= r.mem.total;
/* 把 TP / CP / EP 缩到 N 张卡放得下 */
function fitPar(c, N){ c=Object.assign({},c); while(c.tp*c.cp*(c.pp||1)>N || N%(c.tp*c.cp*(c.pp||1))){ if(c.cp>1) c.cp/=2; else if(c.tp>1) c.tp/=2; else break; }
  while(c.ep>1 && (c.ep>N/(c.pp||1) || c.ep<c.tp)) c.ep/=2; return c; }

/* 代表层的每一段（每层、最忙那张卡）：attention 前后投影、核心、MoE 复制部分、expert、dense FFN、线性注意力 */
const PRE_KEYS=["aqkv","arope","aqa","aqb","akvb","auk","aidxp","aidx"], MOEREP_KEYS=["router","latd","permute","unperm","latu","shared"];
function layerPieces(r){
  const acc = pred => { let f=0,b=0,rc=0; for(const o of r.ops) if(pred(o)){ f+=o.fwd.t/o.layers; b+=(o.dgrad.t+o.wgrad.t)/o.layers; if(o.rc) rc+=o.fwd.t/o.layers; } return {f,b,rc}; };
  const cf=r.par.coreFullOp;
  return { pre:acc(o=>o.group==="attn"&&PRE_KEYS.includes(o.key)), core:cf?acc(o=>o===cf):{f:0,b:0,rc:0}, post:acc(o=>o.group==="attn"&&o.key!=="acore"&&!PRE_KEYS.includes(o.key)),
    rep:acc(o=>MOEREP_KEYS.includes(o.key)), exp:acc(o=>["moe1","act","moe2"].includes(o.key)), den:acc(o=>o.key==="dense"), lin:acc(o=>o.group==="lin") };
}
/* EP 段：torchtitan 直接用 Par.epLayer 的分段；Megatron 从通信表的 a2a 拆成 dispatch / combine */
function epPartsOf(r, P){
  if(r.par.ep) return r.par.ep.parts;
  const {hw,topo}=ctx(), a=r.comm.find(x=>x.key==="a2a"), cn=r.comm.find(x=>x.key==="cnt"), nA=a?+((a.name.match(/×(\d+)/)||[0,4])[1]):4;
  const one=a?a.t/a.layers/nA:0, lane=Common.link(hw,topo,r.ep,r.etp).intra?"nvl":"ib", sync=cn?cn.t/cn.layers:0, f=[], b=[];
  if(sync) f.push(["cpu","计数交换 + D2H 同步",sync,"sync"]);
  f.push([lane,"dispatch",one,"ep"],["sm","expert GEMM",P.exp.f,"moe"],[lane,"combine",one,"ep"]);
  if(nA>4) b.push([lane,"重算：重放 dispatch",one,"ep"],[lane,"重算：重放 combine",one,"ep"]);
  b.push([lane,"combine 的反向",one,"ep"],["sm","expert dgrad + wgrad",P.exp.b+P.exp.rc,"moe"],[lane,"dispatch 的反向",one,"ep"]);
  return {f,b};
}
/* 一层的时间线：which = layer（全注意力层 + MoE / FFN）、attn（只看注意力，比较 CP）、moe（只看 MoE，比较 EP） */
function layerSeq(r, which){
  const {m,d,hw,topo}=ctx(), P=layerPieces(r), ac=r.ops.some(o=>o.rc), f=[], b=[];
  const cp=r.par.cpFull, meth=r.par.cpMeth, C=r.cp, lane=cp&&!cp.lk.intra?"ib":"nvl";
  const cw=cp?r.par.docs*cp.tw:0, cwF=cw*(ac?1/5:1/4), cwB=Math.max(0,cw-cwF*(ac?2:1));
  const tpAll=sum(r.comm.filter(x=>x.cls==="tp"),x=>x.t)/m.L, tpL=Common.link(hw,topo,r.tp,1).intra?"nvl":"ib", H=d.Hattn/r.tp;
  const cpFwd=(arr,coreT,wireT)=>{
    if(C<=1||!cp){ arr.push(["sm","attention 核心",coreT,"attn"]); return; }
    if(meth==="ulysses"){ arr.push([lane,"all-to-all：序列 → 头",wireT/2,"cp"],["sm","attention 核心（"+Math.floor(H/C)+" 个头 × 全序列）",coreT,"attn"],[lane,"all-to-all：头 → 序列",wireT/2,"cp"]); return; }
    if(meth==="allgather"){ arr.push([lane,"all-gather K / V（按 KV 头分块）",wireT,"cp"],["sm","attention 核心（本地 Q × 全部前缀）",coreT,"attn",true]); return; }
    const st=Math.min(C,8); for(let i=0;i<st;i++){ arr.push(["sm","第 "+(i+1)+" 块"+(i===st-1&&C>st?"（共 "+C+" 块）":""),coreT/st,"attn"]); if(i<st-1) arr.push([lane,"P2P 送下一块 K / V",wireT/(st-1),"cp",true]); }
  };
  const moe=m.moe.E>0, EPp=moe?epPartsOf(r,P):null;
  if(which!=="moe"){
    if(tpAll) f.push([tpL,"TP all-gather（SP → 全序列）",tpAll/6,"tp"]);
    f.push(["sm","QKV 投影",P.pre.f,"proj"]); cpFwd(f,P.core.f,cwF); f.push(["sm","O 投影 + 门",P.post.f,"proj"]);
    if(tpAll) f.push([tpL,"TP reduce-scatter",tpAll/6,"tp"]);
  }
  if(which!=="attn"){ if(moe){ f.push(["sm","router / permute / shared",P.rep.f,"proj"]); f.push(...EPp.f); } else f.push(["sm","dense FFN",P.den.f,"proj"]); }
  if(ac && which==="layer") b.push(["sm","重算这一层的前向",P.pre.rc+P.core.rc+P.post.rc+P.rep.rc+(moe?0:P.den.rc),"oth"]);
  if(which!=="attn"){ if(moe){ b.push(...EPp.b); b.push(["sm","router / shared 反向",P.rep.b,"proj"]); } else b.push(["sm","dense FFN 反向",P.den.b,"proj"]); }
  if(which!=="moe"){
    if(tpAll) b.push([tpL,"TP all-gather（反向）",tpAll/2,"tp"]);
    b.push(["sm","O 投影反向",P.post.b,"proj"]);
    const coreB=P.core.b+(which==="attn"?P.core.rc:0);
    if(C<=1||!cp) b.push(["sm","attention 反向",coreB,"attn"]);
    else if(meth==="ulysses") b.push([lane,"all-to-all dO",cwB/2,"cp"],["sm","attention 反向",coreB,"attn"],[lane,"all-to-all dQ / dK / dV",cwB/2,"cp"]);
    else if(meth==="allgather") b.push([lane,"重新 all-gather K / V",cwB/2,"cp"],["sm","attention 反向",coreB,"attn",true],[lane,"reduce-scatter dK / dV",cwB/2,"cp"]);
    else { const st=Math.min(C,8); for(let i=0;i<st;i++){ b.push(["sm","反向第 "+(i+1)+" 块",coreB/st,"attn"]); if(i<st-1) b.push([lane,"P2P：K / V 和 FP32 dK / dV",cwB/(st-1),"cp",true]); } }
    b.push(["sm","QKV 投影反向",P.pre.b,"proj"]);
    if(tpAll) b.push([tpL,"TP reduce-scatter（反向）",tpAll/6,"tp"]);
  }
  return {f,b};
}
const seqGroup = (r, which, label) => { const sq=layerSeq(r,which), F=layout(sq.f,0), B=layout(sq.b,F.end), segs=F.segs.concat(B.segs);
  return { label, lanes:["cpu","sm","nvl","ib"].filter(x=>x==="sm"||segs.some(z=>z.lane===x)), segs, fEnd:F.end, end:B.end }; };

/* ---------- 后训练 · 一个 step：网格与一层时间线 ---------- */
let meshDim="cp", meshSel=0;
function meshDims(r){
  const N=r.dp*r.tp*r.cp*r.pp, Nst=N/r.pp, tp=r.tp, cp=r.cp, ep=r.ep, etp=r.etp, fs=r.par.fsdp, D=[];
  if(tp>1) D.push({k:"tp",n:"TP",c:"c-ar",size:tp,stride:1,g:k=>Math.floor(k/tp)});
  if(cp>1) D.push({k:"cp",n:"CP",c:"c-cp",size:cp,stride:tp,g:k=>Math.floor(k/(tp*cp))*tp+k%tp});
  if(ep>1) D.push({k:"ep",n:"EP",c:"c-a2a",size:ep,stride:etp,g:k=>Math.floor(k/(ep*etp))*etp+k%etp});
  if(fs){ const blk=tp*fs.fs; D.push({k:"fsdp",n:"dense FSDP",c:"c-fsdp",size:fs.fs,stride:tp,g:k=>Math.floor(k/blk)*tp+k%tp});
    if(ep>1 && fs.fe>1){ const eb=ep*fs.fe; D.push({k:"efsdp",n:"expert FSDP",c:"c-moe",size:fs.fe,stride:ep,g:k=>Math.floor(k/eb)*ep+k%ep}); }
    if(fs.R>1){ const rs=Nst/fs.R; D.push({k:"rep",n:"HSDP 副本",c:"c-attn",size:fs.R,stride:rs,g:k=>Math.floor(k/Nst)*rs+k%rs}); } }
  else { if(r.dp>1) D.push({k:"dp",n:"DP（分布式优化器）",c:"c-fsdp",size:r.dp,stride:tp*cp,g:k=>Math.floor(k/Nst)*(tp*cp)+k%(tp*cp)});
    if(ep>1 && r.edp>1) D.push({k:"edp",n:"expert DP",c:"c-moe",size:r.edp,stride:ep*etp,g:k=>Math.floor(k/Nst)*(ep*etp)+k%(ep*etp)}); }
  if(r.pp>1) D.push({k:"pp",n:"PP",c:"c-dp",size:r.pp,stride:Nst,g:k=>k%Nst});
  return D;
}
const PAR_LINK = { tp:"", cp:"#train/par/cp", ep:"#train/par/ep", fsdp:"#train/par/fsdp", efsdp:"#train/par/fsdp", rep:"#train/par/fsdp", dp:"", edp:"", pp:"" };
function renderMesh(r){
  const {hw,topo}=ctx(), N=Math.round(r.dp*r.tp*r.cp*r.pp), dom=Par.pow2Dom(topo), D=meshDims(r);
  if(meshSel>=N) meshSel=0;
  if(!D.length){ $("t-mesh-dim").innerHTML=""; $("t-mesh").innerHTML='<p class="hint">只有一张卡在这个配置里干活。</p>'; $("t-mesh-cap").textContent=""; $("t-groups").innerHTML=""; return; }
  if(!D.some(x=>x.k===meshDim)) meshDim=D[0].k;
  const dim=D.find(x=>x.k===meshDim), selG=dim.g(meshSel);
  $("t-mesh-dim").innerHTML=D.map(x=>'<button type="button" data-v="'+x.k+'" aria-pressed="'+(x.k===meshDim)+'">按 '+esc(x.n)+' 着色</button>').join("");
  let h="";
  for(let lo=0; lo<N; lo+=dom){ h+='<div class="node"><span class="nl">NVLink 域 '+(lo/dom)+(topo.domain>8?"（NVL72 机柜）":"（HGX 节点）")+'</span>';
    for(let k=lo;k<Math.min(N,lo+dom);k++){ const gi=dim.g(k), peer=gi===selG;
      h+='<button type="button" class="g'+(k===meshSel?" sel":"")+(peer?" peer":"")+'" data-k="'+k+'" style="background:'+css(dim.c)+';opacity:'+(peer?1:(gi%2?0.38:0.2))+'" aria-label="GPU '+k+'，'+esc(dim.n)+' 组 '+gi+'">'+(N<=128?k:"")+'</button>'; }
    h+='</div>'; }
  $("t-mesh").innerHTML=h;
  const peers=[]; for(let k=0;k<N;k++) if(dim.g(k)===selG) peers.push(k);
  $("t-mesh-cap").innerHTML='GPU <b>'+meshSel+'</b> 的 '+esc(dim.n)+' 组：'+esc(peers.length>16?peers.slice(0,8).join(", ")+" … "+peers.slice(-4).join(", "):peers.join(", "))+'（'+peers.length+' 张卡）。';
  const cls=k=>r.comm.filter(x=>x.cls===k&&!x.perStage), fs=r.par.fsdp;
  const row=(x,bytes,t,exp,note)=>{ const l=Par.lk(hw,topo,x.size,x.stride); return '<tr><td>'+esc(x.n)+'</td><td class="num">'+x.size+'</td><td class="num">'+x.stride+'</td><td>'+l.kind+'</td><td class="num">'+(bytes!=null?fB(bytes):"—")+'</td><td class="num">'+(t!=null?fT(t):"—")+'</td><td class="num">'+(exp!=null?fT(exp):"—")+'</td><td>'+note+(PAR_LINK[x.k]?' <a href="'+PAR_LINK[x.k]+'">展开 →</a>':'')+'</td></tr>'; };
  const sc=k=>[sum(cls(k),x=>x.bytes),sum(cls(k),x=>x.t),sum(cls(k),x=>x.exp||0)];
  $("t-groups").innerHTML='<thead><tr><th>维度</th><th>组大小</th><th>组内间隔</th><th>链路</th><th>每 micro-batch 每卡字节</th><th>线上时间</th><th>露出</th><th>说明</th></tr></thead><tbody>'+D.map(x=>{
    if(x.k==="tp") return row(x,...sc("tp"),"每层 all-gather / reduce-scatter");
    if(x.k==="cp") return row(x,...sc("cp"),esc(Par.CP[r.par.cpMeth].name)+(r.par.imbCP>1?"，连续切分，最忙 rank "+r.par.imbCP.toFixed(2)+"×":""));
    if(x.k==="ep") return row(x,...sc("ep"),r.par.ep?esc(Par.EP[r.par.ep.be].name)+(r.par.ep.ib>0?"，其中 "+fB(r.nM*r.par.ep.ib)+" 走 IB":""):"Megatron "+esc(S.trn.dispatcher));
    if(x.k==="fsdp") return row(x,fs.bDense,fs.tDense,null,"后台流：每层 all-gather"+(fs.aG===2?"（前向、反向各一次）":"")+" + reduce-scatter；整体露出 "+fT(r.fsdpExposedMB));
    if(x.k==="efsdp") return row(x,fs.bExp,fs.tExp,null,"后台流：每个 MoE 层的 expert 权重");
    if(x.k==="rep") return row(x,null,r.par.tAR,r.tDPexp,"每 step 一次梯度 all-reduce");
    if(x.k==="dp") return row(x,null,r.tDP,r.tDPexp,"每 step 一次：梯度 reduce-scatter + 参数 all-gather");
    if(x.k==="edp") return row(x,null,null,null,"expert 参数的分布式优化器组");
    return row(x,...sc("pp"),"stage 间激活 / 梯度 p2p");
  }).join("")+'</tbody>';
}
function renderLayerGantt(r){
  const {g}=ctx(), G=seqGroup(r,"layer",""), segs=G.segs, fs=r.par.fsdp;
  if(fs && fs.fs>1){ const dL=fs.lkD.intra?"nvl":"ib", P1=Math.max(fs.pFull,fs.pLin);
    const ag=Par.wireT(P1*(fs.fs-1)/fs.fs*2,fs.lkD,g.etaAg,1,g), rs=Par.wireT(P1*(fs.fs-1)/fs.fs*fs.gB,fs.lkD,g.etaAg,1,g);
    segs.push({lane:dL,label:"FSDP 预取下一层（all-gather）",t0:0,t1:ag,cat:"fsdp"});
    if(fs.aG===2) segs.push({lane:dL,label:"FSDP 反向再 gather",t0:G.fEnd,t1:G.fEnd+ag,cat:"fsdp"});
    segs.push({lane:dL,label:"FSDP reduce-scatter 梯度",t0:Math.max(G.fEnd,G.end-rs),t1:G.end,cat:"fsdp"}); }
  if(fs && fs.fe>1 && fs.pExpL){ const eL=fs.lkE.intra?"nvl":"ib"; segs.push({lane:eL,label:"expert FSDP 预取下一层",t0:0,t1:Par.wireT(fs.pExpL*(fs.fe-1)/fs.fe*2,fs.lkE,g.etaAg,1,g),cat:"fsdp"}); }
  G.lanes=["cpu","sm","nvl","ib"].filter(x=>x==="sm"||segs.some(z=>z.lane===x));
  gantt("t-gantt",[G]);
  const {m,d}=ctx();
  $("t-gantt-lede").innerHTML="代表层：一个 "+d.Hattn+" 头的全注意力层"+(m.moe.E?" + MoE":" + dense FFN")+"，最忙那张卡，一个 micro-batch（"+fN(r.Tc)+" token）。前向 "+V(fT(G.fEnd))+"，反向"+(r.ops.some(o=>o.rc)?"（含重算）":"")+" "+V(fT(G.end-G.fEnd))+"。"+
    (fs?"前向开头的绿色段是 FSDP 预取下一层的权重，反向末尾是梯度 reduce-scatter；它们和 SM 上的计算并行，链路被 CP / EP 占着时会互相挤。":"Megatron 的 DP 同步不在层里，在 step 尾部。")+
    ' 各段怎么来的：<a href="#train/par/cp">CP</a> · <a href="#train/par/ep">EP</a> · <a href="#train/par/fsdp">FSDP</a>。';
  legendInto("t-gantt-legend",GANTT_LEGEND.filter(([n,k])=>segs.some(s=>s.cat===k)).map(([n,k])=>({name:n,col:CAT[k]})));
}

/* ---------- “并行”页 ---------- */
const moeLayerT = r => { const P=layerPieces(r), tE=P.exp.f+P.exp.b+P.exp.rc, fs=r.par.fsdp, eF=fs&&r.nM ? fs.tExp*r.pp/r.nM : 0;
  return (r.par.ep ? r.par.ep.t : tE) + Math.max(0, eF - tE); };            // 一层 MoE F+B，加上 expert FSDP 盖不住的部分
const attnLayerT = r => { const P=layerPieces(r), cp=r.par.cpFull; return P.core.f+P.core.b+P.core.rc + (cp ? r.par.docs*cp.exposed : 0); };

function renderPar(){
  const {m,d,hw,topo,g}=ctx(), r=runP(), c=parBase(), mgNow=S.trn.fw==="megatron";
  setWarn(r.warn.concat(runFit(r)?[]:["显存放不下："+fB(r.mem.used)+" > "+hw.mem+" GB"]));
  const fs=r.par.fsdp;
  $("p-lede").innerHTML=(mgNow?'<b>当前框架是 Megatron。</b>这一节讲的是 torchtitan 的 FSDP + CP + EP，所以把同样的 TP '+S.trn.tp+' / CP '+S.trn.cp+' / EP '+S.trn.ep+' 搬过去：PP 去掉，分布式优化器换成 FSDP2，EP 后端换成 '+esc(Par.EP[c.dispatcher].name)+'。在左侧把框架切到 torchtitan，两边就是同一组数。':'')+
    esc(m.short)+" · "+hw.name+" · "+esc(topo.name)+"。每卡每个 micro-batch "+V(fN(r.Tc))+" token（文档长 "+fK(c.S)+"，每组 CP 一个 micro-batch "+fN(r.par.docs)+" 篇）；dense 参数 "+fs.fs+" 路 FSDP"+(fs.R>1?" × "+fs.R+" 个 HSDP 副本":"")+
    (r.cp>1?"，CP "+r.cp+"（"+Par.CP[r.par.cpMeth].name+(r.par.cpMeth!=="ulysses"?(r.par.cpLB?"，causal 均衡":"，连续切分"):"")+"）":"，不开 CP")+
    (m.moe.E?(r.ep>1?"，EP "+r.ep+"（"+Par.EP[r.par.ep.be].name+"）× expert FSDP "+fs.fe:"，不开 EP，expert 跟 dense 一起 FSDP "+fs.fe+" 路"):"，dense 模型没有 expert")+
    "。下面每张图、每个数都是用“一个 step”同一个 <code>Train.run</code> 换参数重跑出来的，改左侧任何参数这里会一起变。";
  kpis($("p-kpis"),[
    ["step 时间",fT(r.stepT),"同“一个 step”"],
    ["吞吐",fN(r.tokPerGpu)+" tok/s/GPU","MFU "+(r.mfu*100).toFixed(1)+"%"],
    ["CP 露出 / micro-batch",fT(sum(r.comm.filter(x=>x.cls==="cp"),x=>x.exp||0)),r.cp>1?"线上 "+fT(sum(r.comm.filter(x=>x.cls==="cp"),x=>x.t)):"CP = 1"],
    ["EP 露出 / micro-batch",fT(sum(r.comm.filter(x=>x.cls==="ep"),x=>x.exp||0)),r.par.ep?"含 host sync "+fT(r.nM*r.par.ep.tSync):"EP = 1"],
    ["FSDP 露出 / micro-batch",fT(r.fsdpExposedMB),"线上 "+fT(r.fsdpMB)+" · 首层 "+fT(fs.firstAG)],
    ["显存",fB(r.mem.used),"上限 "+hw.mem+" GB",!runFit(r)]]);
  renderParRoof(r);
  const N=topo.gpus/(r.tp*r.pp), Nep=topo.gpus/r.pp;
  if(m.moe.E){ const Es=pow2upTo(Nep).filter(v=>m.moe.E%v===0 && v>=r.tp && v>1);
    heat("p-heat-ep","p-heat-ep-legend","EP ",Es,Object.keys(Par.EP),(E,S2,be)=>runP({ep:E,S:S2,tok:Math.max(256,S2/r.cp),dispatcher:be}),k=>Par.EP[k].col,k=>Par.EP[k].short); }
  else { $("p-heat-ep").innerHTML='<text x="340" y="60" text-anchor="middle" class="lbl">dense 模型没有 EP</text>'; $("p-heat-ep-legend").innerHTML=""; }
  const Cs=pow2upTo(Math.min(N,64)).filter(v=>v>1);
  heat("p-heat-cp","p-heat-cp-legend","CP ",Cs,Object.keys(Par.CP),(C,S2,meth)=>runP({cp:C,S:S2,tok:Math.max(256,S2/C),cpType:meth}),k=>Par.CP[k].col,k=>Par.CP[k].name);
  renderParFsdp(r); renderParCp(r); renderParEp(r);
}

/* 三条屋顶线：计算算子按 HBM 强度，通信按“能盖它的 FLOP / 线上字节” */
function renderParRoof(r){
  const {hw,topo}=ctx(), svg=$("p-roof"), W=680, H=360, m={l:64,r:18,t:16,b:44};
  const pk=peakOf(hw,(Train.PREC[S.trn.prec]||Train.PREC.bf16).f);
  const roofs=[{k:"HBM",bw:hw.hbm*1e9,col:"mem"},{k:"NVLink",bw:hw.nvl*1e9,col:"accent"},{k:"IB",bw:hw.ib*1e9,col:"net"}];
  const agg = pred => { let fl=0,by=0,t=0; for(const o of r.ops) if(pred(o)) for(const p of ["fwd","dgrad","wgrad"]){ fl+=o[p].fl; by+=o[p].by; t+=o[p].t; if(p==="fwd"&&o.rc){ fl+=o.fwd.fl; by+=o.fwd.by; t+=o.fwd.t; } } return {fl,by,t}; };
  const groups=[["attention 核心",o=>o.key==="acore","c-attn"],["线性注意力核心",o=>o.key==="lcore","c-lin"],["expert GEMM",o=>["moe1","act","moe2"].includes(o.key),"c-moe"],
    ["投影 / dense / shared GEMM",o=>(o.group==="attn"&&o.key!=="acore")||(o.group==="lin"&&o.key!=="lcore")||["dense","shared","router","latd","latu"].includes(o.key),"c-lm"],
    ["lm_head + CE",o=>o.group==="lm","c-ar"],["norm / permute / 逐元素",o=>o.group==="misc"||["permute","unperm"].includes(o.key),"c-oth"]];
  const ops=groups.map(([name,pred,col])=>Object.assign({name,col},agg(pred))).filter(o=>o.fl>0&&o.by>0&&o.t>0);
  const tot=agg(()=>true), core=agg(o=>o.key==="acore"), ex=agg(o=>["moe1","act","moe2"].includes(o.key)), fs=r.par.fsdp, comm=[];
  const addC=(name,fl,by,lk,col)=>{ if(fl>0&&by>0&&lk) comm.push({name,fl,by,lk,col}); };
  const cpB=sum(r.comm.filter(x=>x.cls==="cp"),x=>x.bytes), epB=sum(r.comm.filter(x=>x.key==="a2a"),x=>x.bytes), tpB=sum(r.comm.filter(x=>x.cls==="tp"),x=>x.bytes);
  if(tpB) addC("TP",tot.fl,tpB,Common.link(hw,topo,r.tp,1),"c-ar");
  if(cpB && r.par.cpFull) addC("CP（"+Par.CP[r.par.cpMeth].name+"）",core.fl,cpB,r.par.cpFull.lk,"c-cp");
  if(epB) addC("EP（"+(r.par.ep?Par.EP[r.par.ep.be].short:S.trn.dispatcher)+"）",ex.fl,epB,r.par.ep?r.par.ep.lk:Common.link(hw,topo,r.ep,r.etp),"c-a2a");
  if(fs && fs.bDense) addC("dense FSDP",tot.fl/r.pp,fs.bDense,fs.lkD,"c-fsdp");
  if(fs && fs.bExp) addC("expert FSDP",ex.fl/r.pp,fs.bExp,fs.lkE,"c-fsdp");
  const X=axis(0.5,1e6,m.l,W-m.r,true), Y=axis(pk/2e4,pk*1.6,H-m.b,m.t,true), clampX=I=>Math.min(Math.max(I,0.6),9e5), clampY=y=>Math.max(y,pk/1.9e4);
  let s="";
  for(const v of [1,10,100,1e3,1e4,1e5,1e6]) s+='<line class="gr" x1="'+X(v)+'" x2="'+X(v)+'" y1="'+m.t+'" y2="'+(H-m.b)+'"/><text x="'+X(v)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+fN(v)+'</text>';
  for(const v of logTicks(pk/2e4,pk*1.6,10)) s+='<line class="gr" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+fN(v/1e12)+'</text>';
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">强度：FLOP / 字节（计算算子按 HBM 字节，通信按线上字节）</text><text x="12" y="'+(m.t-4)+'">TFLOP/s</text>';
  for(const rf of roofs){ const ridge=pk/rf.bw;
    s+='<path d="M'+X(0.5)+','+Y(Math.max(pk/2e4,0.5*rf.bw))+' L'+X(Math.min(ridge,1e6))+','+Y(Math.min(pk,Math.min(ridge,1e6)*rf.bw))+' L'+X(1e6)+','+Y(pk)+'" fill="none" style="stroke:'+css(rf.col)+'" stroke-width="2"/>';
    s+='<line x1="'+X(ridge)+'" x2="'+X(ridge)+'" y1="'+Y(pk)+'" y2="'+(H-m.b)+'" style="stroke:'+css(rf.col)+';stroke-dasharray:3 4;opacity:.55"/>'; }
  for(const o of ops){ const I=o.fl/o.by;
    s+='<circle cx="'+X(clampX(I))+'" cy="'+Y(clampY(Math.min(pk,o.fl/o.t)))+'" r="5.5" style="fill:'+css(o.col)+';stroke:var(--panel)" stroke-width="2" '+tipAttr(o.name,[{v:fN(I)+" FLOP/B",l:"HBM 强度"},{v:fN(o.fl/o.t/1e12)+" TFLOP/s",l:"实际"},{v:fT(o.t),l:"每 micro-batch"}])+'/>'; }
  const badges=[];
  comm.forEach((o,i)=>{ const I=o.fl/o.by, bw=o.lk.bw, y=Math.min(pk,I*bw), xx=X(clampX(I)), yy=Y(clampY(y)); o.no=i+1;
    s+='<path d="M'+xx+','+(yy-7)+' L'+(xx+7)+','+yy+' L'+xx+','+(yy+7)+' L'+(xx-7)+','+yy+' Z" style="fill:var(--panel);stroke:'+css(o.col)+'" stroke-width="2.2" '+tipAttr(o.name,[{v:fN(I)+" FLOP/B",l:"能盖它的计算 / 线上字节"},{v:o.lk.intra?"NVLink":"IB",l:"链路"},{v:I*bw>=pk?"藏得住":"盖不住："+pc(1-I*bw/pk)+" 的时间在等链路",l:""}])+'/>';
    const near=badges.find(b=>Math.hypot(b.x-xx,b.y-yy)<12); if(near) near.t.push(o.no); else badges.push({x:xx,y:yy,t:[o.no]}); });
  for(const b of badges) s+='<text x="'+b.x+'" y="'+(b.y-11)+'" text-anchor="middle" class="lbl" style="pointer-events:none">'+b.t.join(" · ")+'</text>';
  svg.innerHTML=s; bindTips(svg);
  $("p-roof-legend").innerHTML=roofs.map(rf=>'<span><i style="border-color:'+css(rf.col)+'"></i>'+esc(rf.k)+' 屋顶（'+fN(rf.bw/1e9)+' GB/s，拐点 '+fN(pk/rf.bw)+'）</span>').join("")
    +ops.map(o=>'<span><span class="sw" style="background:'+css(o.col)+';border-radius:50%"></span>'+esc(o.name)+'</span>').join("")
    +comm.map(o=>'<span><span class="sw" style="border:2px solid '+css(o.col)+';background:var(--panel);transform:rotate(45deg) scale(.8)"></span>'+o.no+' '+esc(o.name)+'</span>').join("");
}

/* 相图：每格跑一遍 Train.run，颜色是最快的候选 */
const HEAT_S=[8192,16384,32768,65536,131072,262144,524288,1048576];
function heat(svgId, legId, rowLab, rowVals, cands, evalFn, colOf, nameOf){
  const svg=$(svgId), W=680, m={l:62,r:12,t:12,b:40}, rows=rowVals.length, cols=HEAT_S.length, ch=30, H=m.t+m.b+rows*ch;
  svg.setAttribute("viewBox","0 0 "+W+" "+H);
  const cw=(W-m.l-m.r)/cols, used=new Set();
  let s='<defs><pattern id="'+svgId+'-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" style="stroke:var(--line)" stroke-width="2"/></pattern></defs>';
  rowVals.forEach((rv,i)=>{ const y=m.t+(rows-1-i)*ch;
    s+='<text x="'+(m.l-6)+'" y="'+(y+ch/2+4)+'" text-anchor="end">'+esc(rowLab+rv)+'</text>';
    HEAT_S.forEach((S2,j)=>{ const x=m.l+j*cw, res=cands.map(k=>({k,r:evalFn(rv,S2,k)}));
      const ok=res.filter(z=>runOK(z.r)&&runFit(z.r)).sort((a,b)=>b.r.tokPerGpu-a.r.tokPerGpu);
      const tip=res.map(z=>({v:runOK(z.r)?(runFit(z.r)?fN(z.r.tokPerGpu)+" tok/s/GPU":"放不下"):"不可用",l:nameOf(z.k),c:colOf(z.k)}));
      if(ok.length){ const w=ok[0]; used.add(w.k);
        s+='<rect x="'+(x+1)+'" y="'+(y+1)+'" width="'+(cw-2)+'" height="'+(ch-2)+'" rx="3" style="fill:'+css(colOf(w.k))+'" '+tipAttr(rowLab+rv+"，S = "+fK(S2)+"："+nameOf(w.k)+" 最快",tip)+'/>';
        s+='<text x="'+(x+cw/2)+'" y="'+(y+ch/2+4)+'" text-anchor="middle" style="fill:var(--panel);font-size:10.5px;pointer-events:none">'+esc(fN(w.r.tokPerGpu))+'</text>'; }
      else s+='<rect x="'+(x+1)+'" y="'+(y+1)+'" width="'+(cw-2)+'" height="'+(ch-2)+'" rx="3" style="fill:url(#'+svgId+'-hatch)" '+tipAttr(rowLab+rv+"，S = "+fK(S2),tip)+'/>'; }); });
  HEAT_S.forEach((S2,j)=>{ s+='<text x="'+(m.l+j*cw+cw/2)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+fK(S2)+'</text>'; });
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">序列长度 S（每个 micro-batch 一篇）</text>';
  svg.innerHTML=s; bindTips(svg);
  legendInto(legId,cands.map(k=>({name:nameOf(k)+(used.has(k)?"":"（没有赢的格）"),col:colOf(k)})).concat([{name:"斜线：放不下或不合法",col:"line"}]));
}

/* ---------- FSDP ---------- */
function renderParFsdp(r){
  const {m,hw,topo,g}=ctx(), fs=r.par.fsdp, P=layerPieces(r), moe=m.moe.E>0;
  $("p-f-formula").textContent=
"每层每卡线上字节 = P_layer · (F−1)/F · (a·2 + b_g)        a = "+fs.aG+"（reshard_after_forward "+(fs.reshard?"开":"关")+"），b_g = "+fs.gB+"（"+(fs.gB===4?"FP32":"BF16")+" 梯度）\n"+
"dense：P_layer = "+fN(fs.pFull)+"（全注意力层）"+(fs.pLin?" / "+fN(fs.pLin)+"（线性注意力层）":"")+"，F = "+fs.fs+"（"+fs.lkD.kind+"）→ 每 micro-batch "+fB(fs.bDense)+"，"+fT(fs.tDense)+"\n"+
(moe?"expert：P_layer = Z·3·D_e·F_e / EP = "+fN(fs.pExpL)+"，F = "+fs.fe+(fs.fe>1?"（"+fs.lkE.kind+"）→ 每 micro-batch "+fB(fs.bExp)+"，"+fT(fs.tExp):"，没有 expert FSDP 通信")+"\n":"")+
(fs.R>1?"HSDP：每 step 一次 all-reduce 2(R−1)/R × 分片 × b_g，R = "+fs.R+"（"+fs.lkR.kind+"）→ "+fT(r.par.tAR)+"\n":"")+
"每卡常驻状态 = 16 B × 分片参数 "+fN(r.Pshard)+" = "+fB(r.mem.states);
  /* 数据流示意 */
  const Fs=Math.min(fs.fs,8), x0=20, bw=120, gap=26, box=(x,y,w,h,col,txt,op)=>'<rect x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="4" style="fill:'+css(col)+';opacity:'+(op??1)+'"/>'+(txt?'<text x="'+(x+w/2)+'" y="'+(y+h/2+4)+'" text-anchor="middle" style="fill:var(--panel);font-size:10.5px">'+esc(txt)+'</text>':''), lab=(x,y,t)=>'<text x="'+x+'" y="'+y+'" class="lbl">'+esc(t)+'</text>';
  let s=lab(x0,16,"① 常驻：每卡 1/"+fs.fs);
  for(let i=0;i<Fs;i++) s+=box(x0,26+i*24,bw,18,"c-fsdp","GPU "+i+"：分片 "+i,0.9);
  if(fs.fs>8) s+=lab(x0,26+8*24+12,"… 共 "+fs.fs+" 份");
  const x1=x0+bw+gap+40; s+=lab(x1,16,"② all-gather");
  for(let i=0;i<Fs;i++) s+=box(x1,26+i*24,bw,18,"c-lm","",0.85);
  s+='<text x="'+(x1+bw/2)+'" y="'+(26+Math.floor(Fs/2)*24+13)+'" text-anchor="middle" style="fill:var(--panel);font-size:11px">整层 BF16 权重</text>';
  const x2=x1+bw+gap; s+=lab(x2,16,"③ 前向 / 反向计算")+box(x2,26,bw,Fs*24-6,"c-attn","本卡 "+fN(r.Tc)+" token",0.9);
  const x3=x2+bw+gap; s+=lab(x3,16,"④ reduce-scatter 梯度");
  for(let i=0;i<Fs;i++) s+=box(x3,26+i*24,bw-10,18,"c-a2a","∑ 梯度分片 "+i,0.75);
  const yb=26+Fs*24+(fs.fs>8?24:0)+14;     // 说明分两行放在方块下面，图的高度跟着分片数走
  s+='<text x="'+x0+'" y="'+yb+'">每层：前向 gather '+fB(fs.pFull*2)+'，反向'+(fs.reshard?"再 gather 一次":"复用前向的")+'，最后 reduce-scatter '+fB(fs.pFull*fs.gB)+'；</text><text x="'+x0+'" y="'+(yb+16)+'">每卡在线上收发 (F−1)/F 份。Adam 只更新自己那 1/F。</text>';
  for(const xa of [x0+bw,x1+bw,x2+bw]) s+='<path d="M'+(xa+4)+','+(26+Fs*12)+' l'+(gap-10)+',0" style="stroke:var(--muted)" stroke-width="1.5" marker-end="url(#p-f-arr)"/>';
  $("p-f-flow").setAttribute("viewBox","0 0 680 "+(yb+24));
  $("p-f-flow").innerHTML='<defs><marker id="p-f-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 Z" style="fill:var(--muted)"/></marker></defs>'+s;
  /* 连续几层的前向和预取 */
  const ffn=moe?P.rep.f+P.exp.f:P.den.f, compFull=P.pre.f+P.core.f+P.post.f+ffn, compLin=(m.lin.n?P.lin.f/m.lin.n*m.L/m.L:0)+ffn;
  const linPer=m.lin.n? (()=>{ let f=0; for(const o of r.ops) if(o.group==="lin") f+=o.fwd.t/o.layers; return f; })() : 0;
  const lane=fs.lkD.intra?"nvl":"ib", isFull=i=>!m.lin.n||i%4===3, gT=i=>Par.wireT((isFull(i)?fs.pFull:fs.pLin)*(fs.fs-1)/fs.fs*2,fs.lkD,g.etaAg,1,g);
  const layers=Math.min(6,m.L), segs=[]; let commEnd=gT(0), t=commEnd;
  if(commEnd>0) segs.push({lane,label:"gather 第 0 层",t0:0,t1:commEnd,cat:"fsdp"});
  for(let i=0;i<layers;i++){ const comp=isFull(i)?compFull:linPer+ffn, start=Math.max(t,commEnd);
    if(i+1<layers){ const nx=gT(i+1), cs=Math.max(commEnd,start); if(nx>0) segs.push({lane,label:"预取第 "+(i+1)+" 层",t0:cs,t1:cs+nx,cat:"fsdp"}); commEnd=cs+nx; }
    segs.push({lane:"sm",label:"第 "+i+" 层前向"+(isFull(i)?"（全注意力）":"（线性注意力）"),t0:start,t1:start+comp,cat:isFull(i)?"attn":"lin"}); t=start+comp; }
  gantt("p-f-pref",[{label:"",lanes:["sm",lane],segs}]);
  $("p-f-pref-lede").textContent="前 "+layers+" 层的前向：第 i 层计算的同时，链路在 gather 第 i+1 层。全注意力层前向约 "+fT(compFull)+(m.lin.n?"，线性注意力层约 "+fT(linPer+ffn):"")+"；每层 gather 约 "+fT(gT(m.lin.n?3:0))+"（"+fs.lkD.kind+"）。gather 比计算长时，SM 上会出现空档，这就是 FSDP 的露出。"+(fs.fs<=1?"当前 dense FSDP 度是 1，没有 gather。":"");
  legendInto("p-f-pref-legend",[{name:"全注意力层",col:CAT.attn},{name:"线性注意力层",col:CAT.lin},{name:"FSDP all-gather",col:CAT.fsdp}]);
  /* 每卡 token：两种规模各跑一遍 Train.run */
  const gb=/^GB/.test(S.hw), scen = gb ? [["nvl64","64 卡，一个 NVL72（FSDP 在 NVLink 内）","s1"],["nvl512","512 卡，8 个机柜（FSDP 跨柜走 IB）","s2"]] : [["hgx8","8 卡，一台 HGX（FSDP 在 NVLink 内）","s1"],["hgx64","64 卡，8 台 HGX（FSDP 跨机走 IB）","s2"]];
  const ns=[512,1024,2048,4096,8192,16384,32768,65536,131072];
  const ser=scen.map(([tk,name,col])=>({name,short:name.split("，")[0],col,pts:ns.map(n=>{ const c2=fitPar(Object.assign(parBase(),{tok:n,S:Math.max(n*S.trn.cp,1024)}),TOPO[tk].gpus); const r2=Train.run(m,hw,TOPO[tk],Object.assign({N:TOPO[tk].gpus},c2),g);
    return [n, runOK(r2) ? r2.fsdpExposedMB/r2.tMB : null, {hollow:!runFit(r2)}]; })}));
  lineChart("p-f-n",{series:ser,xLog:true,xBase:2,xFmt:fK,yFmt:pc,xLab:"每卡每 micro-batch token",yLab:"FSDP 露出 / micro-batch 时间",vlines:[{x:r.Tc,label:"当前"}],yMin:0});
  legendInto("p-f-n-legend",ser.map(se=>({name:se.name,col:se.col,line:true})));
  const pkb=peakOf(hw,"bf16")*g.etaC, rN=pkb/(hw.nvl*1e9*g.etaAg), rI=pkb/(hw.ib*1e9*g.etaAg);
  $("p-f-n-lede").textContent="FSDP 每层的字节和 token 数无关，计算和 token 数成正比，所以存在一个每卡 token 下限 n*。只看 GEMM：每个参数每步要 6·n（重算 8·n）FLOP，线上要 (a·2 + b_g)·(F−1)/F 字节，n* ≈ (a·2 + b_g) / 6 × 峰值 / 带宽。"+hw.name+" 上 NVLink 的“峰值 / 带宽”约 "+fN(rN)+"，IB 约 "+fN(rI)+"：同样的 FSDP 跨机后要多 "+fN(rI/rN)+" 倍的 token 才能藏住。下图两条线是同一套配置分别放到两种规模上（TP / CP / EP 不够分时自动缩小），露出包括第一层的 gather 和链路被 CP / EP 占用的部分；空心点是显存放不下。";
  $("p-f-tbl").innerHTML='<thead><tr><th>层类型</th><th>层数</th><th>每层每卡参数</th><th>分片组</th><th>每层 FSDP 字节</th><th>每层线上时间</th><th>每层前向计算</th></tr></thead><tbody>'+
    [["全注意力"+(moe?" + MoE 复制部分":""),m.full.n,fs.pFull,fs.fs,fs.lkD,compFull],...(m.lin.n?[["线性注意力"+(moe?" + MoE 复制部分":""),m.lin.n,fs.pLin,fs.fs,fs.lkD,linPer+ffn]]:[]),...(m.dense.n?[["dense FFN",m.dense.n,fs.pDen,fs.fs,fs.lkD,P.den.f]]:[]),...(moe?[["expert（本卡 1/EP）",r.nM,fs.pExpL,fs.fe,fs.lkE,P.exp.f]]:[])]
    .map(([k,nL,Pp,F,l,cf])=>{ const by=Par.fsdpBytes(Pp,F,fs.aG,fs.gB); return '<tr><td>'+esc(k)+'</td><td class="num">'+nL+'</td><td class="num">'+fN(Pp)+'</td><td class="num">'+F+(F>1?"（"+l.kind+"）":"")+'</td><td class="num">'+(F>1?fB(by):"—")+'</td><td class="num">'+(F>1?fT(Par.fsdpT(Pp,F,l,g,fs.aG,fs.gB)):"—")+'</td><td class="num">'+fT(cf)+'</td></tr>'; }).join("")+'</tbody>';
}

/* ---------- CP ---------- */
function cpView(r){ const {topo}=ctx(); if(r.cp>1) return {};
  const C=Math.min(8, topo.gpus/(r.tp*r.pp)); return { cp:C, tok:Math.max(256,S.trn.S/C) }; }
function renderParCp(r){
  const {m,d,topo}=ctx(), V0=cpView(r), rv=Object.keys(V0).length?runP(V0):r, C=rv.cp, c=parBase();
  /* causal 块矩阵 */
  const Cv=Math.min(Math.max(C,2),16), size=280, x0=30, y0=24, cell=size/(2*Cv), uly=rv.par.cpMeth==="ulysses";
  let s='<text x="'+x0+'" y="14" class="lbl">连续切分</text><text x="'+(x0+size+60)+'" y="14" class="lbl">'+(uly?"Ulysses：按头切，每张卡算全部块":"首尾配对：2C 段")+'</text>';
  const work={cont:new Array(Cv).fill(0),zig:new Array(Cv).fill(0)}, rk=(i,zig)=>zig?(i<Cv?i:2*Cv-1-i):Math.floor(i/2);
  for(const [ox,zig] of [[x0,false],[x0+size+60,true]]) for(let i=0;i<2*Cv;i++) for(let j=0;j<=i;j++){ const k=rk(i,zig), diag=i===j; work[zig?"zig":"cont"][k]+=diag?0.5:1;
    s+='<rect x="'+(ox+j*cell)+'" y="'+(y0+i*cell)+'" width="'+(cell-1)+'" height="'+(cell-1)+'" style="fill:'+css(uly&&zig?"c-attn":"s"+(k%6+1))+';opacity:'+(diag?0.55:0.9)+'" '+tipAttr("查询段 "+i+"，键段 "+j,[{v:"rank "+k,l:zig?"首尾配对":"连续"}])+'/>'; }
  $("p-c-mat").innerHTML=s; bindTips($("p-c-mat"));
  const mx=Math.max(...work.cont), avg=a=>a.reduce((x,y)=>x+y,0)/Cv, bh=Math.min(16,270/Cv-3);
  let s2='<text x="20" y="14" class="lbl">每个 rank 的块数（对角块按半块计）</text>';
  for(let k=0;k<Cv;k++){ const y=28+k*(bh+3);
    s2+='<text x="54" y="'+(y+bh-3)+'" text-anchor="end">rank '+k+'</text><rect x="60" y="'+y+'" width="'+(work.cont[k]/mx*260)+'" height="'+(bh/2-1)+'" rx="2" style="fill:var(--s2)" '+tipAttr("rank "+k,[{v:work.cont[k].toFixed(1),l:"连续切分",c:"s2"},{v:work.zig[k].toFixed(1),l:"首尾配对",c:"s1"}])+'/><rect x="60" y="'+(y+bh/2)+'" width="'+(work.zig[k]/mx*260)+'" height="'+(bh/2-1)+'" rx="2" style="fill:var(--s1)"/>'; }
  s2+='<text x="340" y="40" class="lbl">最忙 / 平均</text><text x="340" y="58">连续：'+(mx/avg(work.cont)).toFixed(3)+'（公式 2 − 1/C = '+(2-1/Cv).toFixed(3)+'）</text><text x="340" y="76">首尾配对：'+(Math.max(...work.zig)/avg(work.zig)).toFixed(3)+'</text><text x="340" y="94">Ulysses：1.000（每张卡全部块、1/C 的头）</text>';
  s2+='<rect x="340" y="110" width="12" height="8" style="fill:var(--s2)"/><text x="358" y="118">连续切分</text><rect x="340" y="126" width="12" height="8" style="fill:var(--s1)"/><text x="358" y="134">首尾配对</text>';
  $("p-c-work").innerHTML=s2; bindTips($("p-c-work"));
  $("p-c-mat-lede").textContent="把序列切成 "+(2*Cv)+" 段画出 causal 的下三角（C = "+Cv+(C>16?"，图只画到 16":"")+"）。连续切分时 rank k 拿第 2k、2k+1 段，越靠后的 rank 要看越长的前缀，最后一个 rank 的工作量是平均的 2 − 1/C 倍；首尾配对让 rank k 拿第 k 段和第 2C−1−k 段，每个 rank 正好一样多。Ulysses 不切序列，按头切，天然均衡。"+(S.trn.fw==="megatron"?"Megatron 的 CP 总是首尾配对。":"左侧的“causal 负载均衡”现在"+(c.cpLB!==false?"开着":"关着")+"，最忙 rank 系数 "+rv.par.imbCP.toFixed(3)+"。");
  /* 三种方法 */
  const runs=Object.keys(Par.CP).map(k=>({k,r:runP(Object.assign({},V0,{cpType:k}))}));
  gantt("p-c-gantt",runs.map(({k,r:r2})=>seqGroup(r2,"attn",Par.CP[k].name+"："+fT(layerSeq(r2,"attn")&&seqGroup(r2,"attn","").end)+(r2.par.cpFull&&!r2.par.cpFull.ok?"（"+r2.par.cpFull.why+"）":""))));
  legendInto("p-c-gantt-legend",[["attention","attn"],["投影","proj"],["CP 通信","cp"],["TP 通信","tp"]].map(([n,k])=>({name:n,col:CAT[k]})));
  $("p-c-gantt-lede").textContent=(Object.keys(V0).length?"当前 CP = 1，这里按 CP "+C+" 演示（每卡 "+fN(rv.Tc)+" token）。":"")+"一个全注意力层（不含 FFN），最忙那张卡，一个 micro-batch 的前向 + 反向（重算也算进反向）。Ring 的 P2P 和分块计算交替，块越多越能藏；AllGather 先拿全部 K / V，按 KV 头分块流水；Ulysses 前后各一次 all-to-all，中间是一次完整的 attention。";
  const ac=rv.ops.some(o=>o.rc), kv=d.kvTokElems, tbl=$("p-c-tbl");
  tbl.innerHTML='<thead><tr><th>方法</th><th>F+B 每卡发出（'+(ac?"full 重算":"不重算")+'）</th><th>一层、一个 micro-batch</th><th>调用次数 / 篇</th><th>最忙 rank 工作量</th><th>线上 / 露出</th></tr></thead><tbody>'+
    runs.map(({k,r:r2})=>{ const cp=r2.par.cpFull; if(!cp) return ""; const form=k==="allgather"?(ac?4:3)+"(C−1)·L":k==="ring"?(ac?"(5C−3)·L":"(4C−2)·L"):(ac?6:4)+"·b·(S/C)·(Q+K′)·d·(C−1)/C";
      return '<tr'+(k===rv.par.cpMeth?' class="hl"':'')+'><td>'+esc(Par.CP[k].name)+'</td><td><code>'+esc(form)+'</code></td><td class="num">'+fB(r2.par.docs*cp.wire)+'</td><td class="num">'+cp.calls+'</td><td class="num">'+r2.par.imbCP.toFixed(3)+'×</td><td class="num">'+fT(r2.par.docs*cp.tw)+' / '+fT(r2.par.docs*cp.exposed)+(cp.ok?'':' <span class="tag">'+esc(cp.why)+'</span>')+'</td></tr>'; }).join("")+
    '<tr><td colspan="6">L = 本卡一份 K + V 的字节 = 2 B × S/C × '+kv+'（'+(m.full.kind==="gqa"?"2 × "+m.full.Hkv+" 个 KV 头 × "+m.full.dh:"latent + rope"+(m.full.kind==="dsa"?"，另加 indexer key "+m.full.idxD:""))+'），按 TP 切 KV 头。Ulysses 的 K′ = max(Hkv, C)'+(m.full.kind==="gqa"?" = "+Math.max(m.full.Hkv,C):"；MLA 的 latent 不能按头切，按 C 份复制")+'。</td></tr></tbody>';
  /* 随 C、随 S、固定世界 */
  const N=topo.gpus/(r.tp*r.pp), Cs=pow2upTo(Math.min(64,N)).filter(v=>v>1);
  const serC=Object.keys(Par.CP).map(k=>({name:Par.CP[k].name,col:Par.CP[k].col,pts:Cs.map(Cx=>{ const r2=runP({cpType:k,cp:Cx,tok:Math.max(64,S.trn.S/Cx)}); return [Cx, runOK(r2)&&r2.par.cpFull ? attnLayerT(r2) : null]; })}));
  lineChart("p-c-vsC",{series:serC,xLog:true,xBase:2,xLab:"CP 度 C",yLab:"一层 attention F+B（核心 + 露出通信）",yLog:true,yFmt:fT});
  legendInto("p-c-vsC-legend",serC.map(se=>({name:se.name,col:se.col,line:true})));
  const Ss=[8192,16384,32768,65536,131072,262144,524288,1048576];
  const serS=Object.keys(Par.CP).map(k=>({name:Par.CP[k].name,col:Par.CP[k].col,pts:Ss.map(S2=>{ const r2=runP(Object.assign({},V0,{cpType:k,S:S2,tok:Math.max(256,S2/C)})); return [S2, runOK(r2)?r2.tokPerGpu:null, {hollow:!runFit(r2),note:runFit(r2)?"":"（放不下）"}]; })}));
  lineChart("p-c-vsS",{series:serS,xLog:true,xBase:2,xFmt:fK,xLab:"序列长度 S",yLab:"单卡 tok/s（空心：放不下）",vlines:[{x:S.trn.S,label:"当前"}]});
  legendInto("p-c-vsS-legend",serS.map(se=>({name:se.name,col:se.col,line:true})));
  const S0=Math.min(S.trn.S,32768), Cf=pow2upTo(Math.min(64,N));
  const serF=Object.keys(Par.CP).map(k=>({name:Par.CP[k].name,col:Par.CP[k].col,pts:Cf.map(Cx=>{ const r2=runP({cpType:k,cp:Cx,S:S0,tok:S0}); return [Cx, runOK(r2)?r2.stepT:null, {hollow:!runFit(r2)}]; })}));
  lineChart("p-c-fixed",{series:serF,xLog:true,xBase:2,xLab:"CP 度 C（卡数、每步 token 不变）",yLab:"一步时间（空心：放不下）",yFmt:fT});
  legendInto("p-c-fixed-legend",serF.map(se=>({name:se.name,col:se.col,line:true})));
  $("p-c-fixed-lede").textContent="卡数和每步 token 不变，文档长 "+fK(S0)+"，每卡每个 micro-batch 也是 "+fK(S0)+" token（CP = C 时一组 C 张卡一起处理 C 篇，每篇切成 C 份），只改 CP 度。总 FLOP 不变，多出来的只有 CP 通信和不均衡。"+fixedWorldNote()+"CP 的价值是让一篇放不下的文档放得下、或者在文档少于数据通道时把空闲的卡用起来，不是让固定的一批算得更快。";
}

/* 匿名实测（数据点库）给两段说明配的数 */
function fixedWorldNote(){ const at=(meth,cp)=>DB.one("cp_fixed_world",{model:"qwen35",seq:32768,method:meth,cp}); const b=at("single",1);
  if(!b) return ""; const ms=p=>p?Math.round(p.fb_ms)+" ms":"—";
  return "匿名数据点：8 × H200、8 篇 32K 文档时 CP1 / DP8 的 attention 是 "+ms(b)+"，CP8 Ulysses "+ms(at("ulysses",8))+"、AllGather "+ms(at("allgather",8))+"、Ring "+ms(at("ring",8))+"。"; }
function crossNote(){ const p=DB.one("efsdp_crossover",{case:"same_node"}); return p ? "匿名数据点：8 × H200 上 Qwen3.5 的 EP4 → EP8 交叉约 "+fInt(p.tokens)+" × B_E / B_F 个 token。" : ""; }

/* ---------- EP ---------- */
const EP_CARDS = {
  standard:{ h:"NCCL all-to-all（torchtitan 默认）", p:"本地按 expert 排序 → 先 all-to-all 交换每个 expert 的计数 → D2H 把 split 拷到 CPU（阻塞）→ all_to_all_single 交换 token → 本地 permute 成 expert-major → GEMM → 反向过程。字节最少（只发需要的槽位），但每层一次 host sync，CPU 跑不到 GPU 前面，也做不了 CUDA graph；full 重算时还要重放一次。", m:"torchtitan AllToAllTokenDispatcher" },
  ag:{ h:"AllGather + ReduceScatter（NCCL）", p:"每张卡把自己的 n 个 token 全部广播给 EP 组（all-gather），每张卡只挑命中本地 expert 的行去算，再用 reduce-scatter 把各卡的部分结果加回原主。形状完全静态：不用知道 split、不用 host sync、能进 CUDA graph。代价是每卡收 (E−1)·n 行，而 a2a 只收 n·K·(1−1/E) 行：比值是 E/K，E ≤ K 时它反而不比 a2a 多。", m:"Megatron allgather dispatcher · vLLM 默认 allgather_reducescatter" },
  deepep:{ h:"DeepEP v2 expanded", p:"kernel 发起的 push：发送方把每个 token 按目的 rank 去重后写进对端的对称接收区，接收方在 copy epilogue 里直接展开成 expert-major 的行，省掉单独的 pack / fold。训练用 do_cpu_sync=True 拿精确大小（每层一次 CPU 轮询）；不同步就按容量预留。能跨机（NVLink + NCCL GIN），按节点去重。注意：pinned 的 torchtitan 适配器在 autograd 下强制 compact，expanded 的训练反向需要自己实现。", m:"DeepEP ElasticBuffer / EPBuffer · torchtitan DeepEPTokenDispatcher" },
  distmoe:{ h:"torch Dist-MoE", p:"pull + 零拷贝：每张卡只把自己的 x 和 route id 写进自己的对称 buffer，barrier 之后，每个接收方读所有 peer 的 route id、在本地算出指针表；W13 grouped GEMM 的 gather warp 按指针直接从对端显存 TMA 读行，W2 的 epilogue 直接写进 token 原主的 combine buffer。all-to-all 这一步消失了，通信藏在 GEMM 的 producer 和 epilogue 里。CPU 从不知道大小，整层能进 CUDA graph，行序确定。只支持 SM100+、只能在 NVLink 域内。", m:"meta-pytorch/dist_moe · torchtitan models/common/dist_moe" },
  mok:{ h:"Mixture-of-Kittens", p:"把 dispatch、expert FFN、combine 全部放进一个持久 megakernel：一部分 SM 专做通信（pull dispatch、push combine，不到 1/3 的 SM 就能打满 NVLink），其余 SM 做 grouped GEMM 和 SwiGLU，彼此用本地计数器握手；token 以 minibatch 流过一个几百 MB 的环形 buffer，不需要精确大小，也没有 CPU 同步。浮点运算顺序固定，输出逐位确定；wgrad 在所有 dgrad 之后沿整个 token 轴一次累加。MXFP8 量化融进 dispatch。面向 GB200 / GB300 NVL72。左侧“更多”里的“MoK 通信 SM 比例”就是这里的 f。", m:"cursor/mixture-of-kittens（Apache-2.0，2026-08）" } };
function epView(r){ const {topo,m}=ctx(); if(r.ep>1) return {}; const N=topo.gpus/r.pp; let E=Math.min(8,N); while(E>1 && m.moe.E%E) E/=2; return { ep:E }; }
function renderParEp(r){
  const {m,d,hw,topo,g}=ctx(), e=m.moe, ids=["p-e-cards","p-e-props","p-e-gantt","p-e-vsE","p-e-vsN","p-e-cross","p-e-fan"];
  if(!e.E){ $("p-e-formula").textContent="当前是 dense 模型，没有 expert。换一个 MoE 模型看这一节。"; ids.forEach(id=>{ const el=$(id); el.innerHTML=el.tagName==="svg"?'<text x="340" y="40" text-anchor="middle" class="lbl">dense 模型没有 EP</text>':""; });
    ["p-e-gantt-legend","p-e-vsE-legend","p-e-vsN-legend","p-e-cross-legend"].forEach(id=>$(id).innerHTML=""); $("p-e-gantt-lede").textContent=""; $("p-e-cross-lede").textContent=""; return; }
  const V0=epView(r), rv=Object.keys(V0).length?runP(V0):r, n=rv.Tl, K=e.k, Z=e.E, E=rv.ep, De=d.De, pE=Par.pDest(Z,E,K);
  $("p-e-formula").textContent=(Object.keys(V0).length?"当前 EP = 1，这一节按 EP "+E+" 演示。\n":"")+
"当前：每卡 n = "+fN(n)+" token，top-"+K+" / "+Z+" expert，EP"+E+"，dispatch 宽度 D_e = "+De+(e.Dlat?"（latent）":"")+"\n"+
"按 expert 槽位：n·K·(1−1/E) = "+fN(n*K*(1-1/E))+" 行            每层每次 "+fB(n*K*(1-1/E)*De*2)+"\n"+
"按目的 rank 去重：n·(E−1)·p(E) = "+fN(n*(E-1)*pE)+" 行   p(E) = "+pE.toFixed(3)+"   "+fB(n*(E-1)*pE*De*2)+"\n"+
"整份广播（AllGather）：(E−1)·n = "+fN((E-1)*n)+" 行                       "+fB((E-1)*n*De*2)+"\n"+
"接收端 expanded / compact 行数比 = K / (E·p(E)) = "+(K/(E*pE)).toFixed(2);
  $("p-e-cards").innerHTML=Object.keys(Par.EP).map(k=>'<div class="card"><h4><span class="dot" style="background:'+css(Par.EP[k].col)+'"></span>'+esc(EP_CARDS[k].h)+'</h4><p>'+esc(EP_CARDS[k].p)+'</p><p class="meta">'+esc(EP_CARDS[k].m)+'</p></div>').join("");
  const all=Object.keys(Par.EP).map(k=>({k,r:runP(Object.assign({},V0,{dispatcher:k}))}));
  $("p-e-props").innerHTML='<thead><tr><th>后端</th><th>谁发起</th><th>每层 F+B 线上字节</th><th>线上时间</th><th>一层 MoE F+B</th><th>host sync / 层</th><th>buffer / 卡</th><th>确定性</th><th>能跨 NVLink 域</th><th>当前可用</th><th></th></tr></thead><tbody>'+
    all.map(({k,r:r2})=>{ const ep=r2.par.ep; return '<tr'+(k===rv.par.ep.be?' class="hl"':'')+'><td>'+esc(Par.EP[k].name)+'</td><td>'+(k==="standard"||k==="ag"?"host（NCCL）":"kernel")+'</td><td class="num">'+fB(ep.wire)+'</td><td class="num">'+fT(ep.tw)+'</td><td class="num">'+fT(ep.t)+'</td><td class="num">'+ep.syncs+'</td><td class="num">'+fB(ep.buf)+'</td><td>'+(ep.det?"是":"默认否")+'</td><td>'+(Par.EP[k].fused?"否":"是")+'</td><td class="'+(ep.ok?"ok":"bad")+'">'+(ep.ok?"是":esc(ep.why))+'</td><td>'+(S.trn.fw==="torchtitan"&&k!==S.trn.dispatcher?'<button type="button" class="mini" data-disp="'+k+'">用这个</button>':'')+'</td></tr>'; }).join("")+'</tbody>';
  gantt("p-e-gantt",all.map(({k,r:r2})=>seqGroup(r2,"moe",Par.EP[k].name+"："+fT(seqGroup(r2,"moe","").end)+(r2.par.ep.ok?"":"（不可用："+r2.par.ep.why+"）"))));
  legendInto("p-e-gantt-legend",[["MoE GEMM","moe"],["EP 通信","ep"],["router / permute / shared","proj"],["同步 / barrier","sync"]].map(([nm,k])=>({name:nm,col:CAT[k]})));
  $("p-e-gantt-lede").textContent="一层 MoE 的前向 + 反向，最忙那张卡（每卡 "+fN(n)+" token，路由不均衡 "+S.trn.imb+"）。前三个是“先通信、再计算、再通信”的串行结构，差别在字节和有没有 host sync；Dist-MoE 和 MoK 的通信段和 GEMM 段在时间上重叠，层时间取两者中较长的那个。";
  const N=topo.gpus/r.pp, Es=pow2upTo(N).filter(v=>Z%v===0 && v>=rv.tp);
  const serE=Object.keys(Par.EP).map(k=>({name:Par.EP[k].name,short:Par.EP[k].short,col:Par.EP[k].col,pts:Es.map(Ex=>{ const r2=runP({ep:Ex,dispatcher:k}); return [Ex, runOK(r2)?moeLayerT(r2):null]; })}));
  lineChart("p-e-vsE",{series:serE,xLog:true,xBase:2,xLab:"EP 度",yLab:"一层 MoE F+B",yFmt:fT,yLog:true,vlines:N>Par.pow2Dom(topo)?[{x:Par.pow2Dom(topo),label:"NVLink 域"}]:[]});
  legendInto("p-e-vsE-legend",serE.map(se=>({name:se.name,col:se.col,line:true})));
  const ns=[1024,2048,4096,8192,16384,32768,65536,131072];
  const serN=Object.keys(Par.EP).map(k=>({name:Par.EP[k].name,short:Par.EP[k].short,col:Par.EP[k].col,pts:ns.map(nx=>{ const r2=runP(Object.assign({},V0,{dispatcher:k,tok:nx*rv.tp})); return [nx, runOK(r2)?moeLayerT(r2)/nx*1e6:null]; })}));
  lineChart("p-e-vsN",{series:serN,xLog:true,xBase:2,xFmt:fK,xLab:"每卡 token 数 n",yLab:"一层 MoE F+B，每百万 token",yFmt:v=>fT(v),vlines:[{x:n,label:"当前"}]});
  legendInto("p-e-vsN-legend",serN.map(se=>({name:se.name,col:se.col,line:true})));
  /* 交叉：expert FSDP 权重字节对激活字节 */
  const fs=rv.par.fsdp, blk=rv.ep*fs.fe, rep=rv.ops.some(o=>o.rc);
  const Wb=Ex=>{ const F=blk/Ex; return F>=1 ? rv.nM*Z*3*De*e.F/Ex*(F-1)/F*(fs.aG*2+fs.gB) : null; };
  const Vb=(Ex,kind)=>{ const mult=4+(rep&&kind!=="dedup"?2:0); return rv.nM*mult*n*De*2*(kind==="slot"?K*(1-1/Ex):kind==="dedup"?(Ex-1)*Par.pDest(Z,Ex,K):(Ex-1)); };
  const Eb=Es.filter(x=>x<=blk);
  const serX=[{name:"expert FSDP 权重 + 梯度",short:"权重",col:"c-fsdp",pts:Eb.map(Ex=>[Ex,Wb(Ex)||null])},
    {name:"激活：按槽位（a2a / Dist-MoE / MoK）",short:"按槽位",col:"s5",pts:Eb.filter(x=>x>1).map(Ex=>[Ex,Vb(Ex,"slot")])},
    {name:"激活：按 rank 去重（DeepEP）",short:"去重",col:"s2",pts:Eb.filter(x=>x>1).map(Ex=>[Ex,Vb(Ex,"dedup")])},
    {name:"激活：整份广播（AllGather）",short:"广播",col:"s1",pts:Eb.filter(x=>x>1).map(Ex=>[Ex,Vb(Ex,"bcast")])}];
  lineChart("p-e-cross",{series:serX,xLog:true,xBase:2,yLog:true,xLab:"EP 度（EP × expert FSDP = "+blk+" 不变）",yLab:"每 micro-batch 每卡字节",yFmt:fB,tipFmt:fB});
  legendInto("p-e-cross-legend",serX.map(se=>({name:se.name,col:se.col,line:true})));
  const Tstar=(()=>{ const a=Math.max(1,E/2), b=E; if(a===b||!Wb(a)) return null; const dW=Wb(a)-(Wb(b)||0), dV=(Vb(b,"slot")-(a>1?Vb(a,"slot"):0))/n; return dV>0?dW/dV:null; })();
  $("p-e-cross-lede").textContent="EP × expert FSDP 固定时，EP 越大，expert 权重的 FSDP 流量越小（expert FSDP = 1 时为 0），token 的激活流量越大。权重字节和 token 数无关，激活字节和 token 数成正比，所以长序列 / 大 micro-batch 偏向小 EP + expert FSDP，短序列偏向大 EP。按槽位计、两条在同一条链路上时，从 EP"+Math.max(1,E/2)+" 换到 EP"+E+" 的交叉点约为每卡 "+(Tstar?fN(Tstar):"—")+" 个 token（当前 "+fN(n)+"）；权重和激活走不同链路（比如 expert FSDP 跨机、EP 在机内）时要再乘两条链路的带宽比。"+crossNote()+"";
  $("p-e-fan").innerHTML='<thead><tr><th>EP</th><th>p(E)：token 命中某 rank</th><th>远端 rank 数 d(E)</th><th>按槽位远端行 K(1−1/E)</th><th>expanded / compact 行数</th><th>每层每次 dispatch（槽位 / 去重 / 广播）</th></tr></thead><tbody>'+
    Es.filter(x=>x>1).map(Ex=>{ const p=Par.pDest(Z,Ex,K); return '<tr'+(Ex===E?' class="hl"':'')+'><td class="num">'+Ex+'</td><td class="num">'+p.toFixed(3)+'</td><td class="num">'+((Ex-1)*p).toFixed(2)+'</td><td class="num">'+(K*(1-1/Ex)).toFixed(2)+'</td><td class="num">'+(K/(Ex*p)).toFixed(2)+'</td><td class="num">'+fB(n*K*(1-1/Ex)*De*2)+' / '+fB(n*(Ex-1)*p*De*2)+' / '+fB((Ex-1)*n*De*2)+'</td></tr>'; }).join("")+'</tbody>';
}

