/* ---------- 一层里的通信：先后顺序和每卡字节（后训练 · 并行、推理 · 部署各一张） ----------
   行按发出的先后排。左边是本卡的计算流，右边每个并行维度一条道；通信挂在“等它”或“和它同时跑”的那段计算的同一行。
   实心格是阻塞的：这一行的计算要等它做完才开始。虚线格在后台流上跑，被这一行的计算盖住。
   每个格子的字节和时间都取自 Train.run / Infer.step 里同一组逐次记录（Par.xc），所以和“一个 step”的通信表加起来一致。 */
const XD = { tp:["TP","c-ar"], cp:["CP","c-cp"], ep:["EP","c-a2a"], etp:["ETP","c-moe"], fsdp:["FSDP","c-fsdp"], efsdp:["expert FSDP","c-moe"],
  pp:["PP","c-dp"], dp:["DP","c-fsdp"], edp:["expert DP","c-moe"], rep:["HSDP 副本","c-attn"], dpa:["DP-attention","c-dp"] };
const XORDER = ["tp","dpa","cp","ep","etp","fsdp","efsdp","pp","dp","edp","rep"];
const XOP = { AG:"all-gather", RS:"reduce-scatter", AR:"all-reduce", A2A:"all-to-all", P2P:"send / recv", sync:"host 同步" };
const XMODE = { block:"阻塞：这一行的计算等它做完", bg:"后台流：和这一行的计算同时跑", part:"分块重叠：一部分露出", fused:"融合在计算 kernel 里" };
const XV = { trn:{ n:2, show:"both" }, inf:{ n:2, cfg:"cur" } };

/* 事件 → 行：每段计算开一行，前面攒下的通信挂到这一行（阻塞的是它要等的，后台的是和它同时跑的）；
   没有通信隔开的相邻计算合成一行。一层结束时还没被等到的阻塞通信单独成一行“等通信”，后台的留给下一层 */
function xRows(ev){
  const rows=[]; let pend=[], last=null;
  const flushWait=()=>{ const b=pend.filter(x=>x.mode==="block"); if(b.length){ rows.push({comp:null,chips:b}); pend=pend.filter(x=>x.mode!=="block"); } };
  for(const e of ev){
    if(e.k==="s"){ flushWait(); rows.push({sec:e}); last=null; continue; }
    if(e.k==="x"){ pend.push(e); continue; }
    if(!(e.t>0) && !e.ghost) continue;
    if(last && !pend.length && !e.ghost && !e.solo){ last.comp=Object.assign({},last.comp,{label:last.comp.label+" → "+e.label,t:last.comp.t+e.t}); continue; }
    const row={comp:e,chips:pend}; pend=[]; rows.push(row); last=e.ghost||e.solo?null:row;
  }
  if(pend.length) rows.push({comp:null,chips:pend});
  return rows;
}
const xMembers = (n, stride) => { const a=[]; for(let i=0;i<n;i++) a.push(i*stride); return a.length>8 ? a.slice(0,4).join(", ")+" … "+a.slice(-2).join(", ") : a.join(", "); };
function xGrid(elId, ev, groups){
  const rows=xRows(ev), lanes=XORDER.filter(k=>ev.some(e=>e.k==="x"&&e.dim===k)), el=$(elId);
  if(!lanes.length){ el.style.setProperty("--k",1); el.innerHTML='<div class="xs">这个配置一层里没有集合通信。</div>'; return lanes; }
  const mx=Math.max(1,...ev.filter(e=>e.k==="x").map(e=>e.bytes));
  let n=0, h='<div class="xr xh"><div class="xn">#</div><div class="xc0">本卡计算</div>'+lanes.map(k=>{ const G=groups[k]||{};
    return '<div class="xl" style="--c:'+css(XD[k][1])+'"><b>'+esc(XD[k][0])+'</b> '+(G.n||"")+' 卡'+(G.lk?' · '+esc(G.lk):'')+(G.mem?'<span>GPU 0 的组：'+esc(G.mem)+'</span>':'')+'</div>'; }).join("")+'</div>';
  for(const r of rows){
    if(r.sec){ h+='<div class="xs'+(r.sec.sub?' sub':'')+'">'+r.sec.label+'</div>'; continue; }
    const c=r.comp;
    h+='<div class="xr"><div class="xn">'+(++n)+'</div>'+(c ? '<div class="xc0'+(c.ghost?' ghost':'')+'" style="--cc:'+css(CAT[c.cat]||"c-oth")+'"><span>'+esc(c.label)+'</span>'+(c.t>0?'<span class="t">'+fT(c.t)+'</span>':'')+'</div>'
      : '<div class="xc0 wait"><span>等通信</span></div>');
    for(const k of lanes) h+='<div class="xl">'+r.chips.filter(x=>x.dim===k).map(x=>xChip(x,mx,groups[k]||{})).join("")+'</div>';
    h+='</div>';
  }
  el.style.setProperty("--k",lanes.length); el.innerHTML=h; bindTips(el);
  return lanes;
}
function xChip(x, mx, G){
  const bw = x.t>0 && x.bytes>0 ? x.bytes/x.t/1e9 : 0;
  const tip=tipAttr(XD[x.dim][0]+" · "+XOP[x.op]+"："+x.what,[{v:fB(x.bytes),l:"每卡发出"},{v:fT(x.t),l:"线上时间"+(bw?"（约 "+fN(bw)+" GB/s）":"")},
    {v:(G.n||"?")+" 卡",l:(G.lk||"")+(G.mem?"，GPU 0 的组 "+G.mem:"")},{v:XMODE[x.mode],l:x.note||""}]);
  return '<div class="xchip '+x.mode+'" style="--c:'+css(XD[x.dim][1])+'" '+tip+'><div class="x1"><b class="xo">'+x.op+'</b><span>'+esc(x.what)+'</span></div>'+
    '<div class="x2"><b>'+fB(x.bytes)+'</b><span>'+fT(x.t)+'</span>'+(x.mode!=="block"?'<em>'+({bg:"后台",part:"分块重叠",fused:"融合"}[x.mode])+'</em>':'')+'<span class="xd">'+esc(XD[x.dim][0])+'</span></div>'+
    '<i class="xbar"><i style="width:'+Math.max(1.5,x.bytes/mx*100).toFixed(1)+'%"></i></i></div>';
}
/* 每个样例层的线上字节，按维度堆叠；同一把尺子 */
function xVolume(elId, items){
  const mx=Math.max(1,...items.map(it=>sum(it.parts,p=>p.v)));
  $(elId).innerHTML=items.map(it=>{ const tot=sum(it.parts,p=>p.v);
    return '<div class="xv"><span class="lab">'+it.label+'</span><span class="bar">'+it.parts.filter(p=>p.v>0).map(p=>'<i style="width:'+(p.v/mx*100).toFixed(2)+'%;background:'+css(XD[p.dim][1])+'" '+tipAttr(it.label.replace(/<[^>]+>/g,""),[{v:fB(p.v),l:XD[p.dim][0]+"（"+pc(p.v/tot)+"）"}])+'></i>').join("")+'</span><span class="num">'+fB(tot)+'</span></div>'; }).join("");
  bindTips($(elId));
}
const xSeg = (elId, opts, cur, on) => { const el=$(elId);
  el.innerHTML=opts.map(([v,l])=>'<button type="button" data-v="'+v+'" aria-pressed="'+(String(v)===String(cur))+'">'+esc(l)+'</button>').join("");
  el.onclick=e=>{ const b=e.target.closest("button"); if(b) on(b.dataset.v); }; };

/* 样例层：取一个全注意力层，和它前面的一两层（混合模型里是线性 / 滑窗层），都落在 MoE 区 */
function xKinds(m){
  const moe=m.moe.E>0, f=m.full;
  const kind = i => ({ a: m.lin.n ? (i%4===3?"full":"lin") : f.swa ? (i%(f.gEvery||4)===(f.gOff||0)?"full":"swa") : "full", f: (moe && i>=m.dense.n) ? "moe" : "dense" });
  const name = k => (k.a==="lin" ? m.lin.kind==="GDN"?"Gated DeltaNet":"KDA" : k.a==="swa" ? "滑窗注意力 "+f.W : f.kind==="gqa" ? (f.swa?"全局注意力":"全注意力") : f.kind.toUpperCase())+" + "+(k.f==="moe"?"MoE":"dense FFN");
  return { kind, name };
}
function xSample(m, n){
  const {kind}=xKinds(m), lo=(m.moe.E?m.dense.n:0)+n-1;
  let i=lo; while(i<m.L && kind(i).a!=="full") i++; if(i>=m.L) i=lo;
  const L=[]; for(let j=i-n+1;j<=i;j++) L.push(j); return L;
}

/* ================= 训练：当前框架、当前配置（和“一个 step”同一个 Train.run） ================= */
function trnPlan(r, L, show){
  const {m,g}=ctx(), c=S.trn, mg=c.fw==="megatron", fs=r.par.fsdp, f=m.full, {kind,name}=xKinds(m);
  const per = pred => { let F=0,B=0,RC=0; for(const o of r.ops) if(pred(o)){ const n=o.layers||1; F+=o.fwd.t/n; B+=(o.dgrad.t+o.wgrad.t)/n; if(o.rc) RC+=o.fwd.t/n; } return {f:F,b:B,rc:RC}; };
  const cf=r.par.coreFullOp, cs=r.ops.find(o=>o.key==="acore"&&o!==cf);
  const PC = { pre:per(o=>o.group==="attn"&&PRE_KEYS.includes(o.key)), full:per(o=>o===cf), swa:per(o=>o===cs), post:per(o=>o.group==="attn"&&o.key!=="acore"&&!PRE_KEYS.includes(o.key)),
    lpre:per(o=>o.key==="lproj"), lmid:per(o=>["lconv","lcore","lnorm"].includes(o.key)), lpost:per(o=>o.key==="lout"),
    router:per(o=>["router","latd","permute"].includes(o.key)), exp:per(o=>["moe1","act","moe2"].includes(o.key)), unp:per(o=>["unperm","latu"].includes(o.key)), shared:per(o=>o.key==="shared"), dense:per(o=>o.key==="dense") };
  const xsOf = (key, blk) => { const e=r.comm.find(x=>x.key===key&&(!blk||x.blk===blk)); return e&&e.xs ? e.xs.map(x=>Object.assign({},x,{ent:e})) : []; };
  const P = (xs, pass) => xs.filter(x=>x.pass===pass);
  const ev=[], C=(label,t,cat,o)=>ev.push(Object.assign({k:"c",label,t,cat},o||{})), sec=(label,o)=>ev.push(Object.assign({k:"s",label},o||{}));
  const X=(dim,x,mode,lay,o)=>{ if(x) ev.push(Object.assign({k:"x",dim,mode,lay},x,o||{})); };
  const tpM = c.tpOverlap ? "part" : "block", tpNote = c.tpOverlap ? "userbuffers 把 AG / RS 切块和 GEMM 交错" : "";
  const epAll = xsOf("a2a","moe"), cnt = xsOf("cnt","moe"), etp = xsOf("etp","moe"), epFused = epAll.length && epAll[0].ent.fused;
  const epM = epFused ? "fused" : (mg && c.epOverlap ? "part" : "block"), epNote = epFused ? "对端 load / store 在 GEMM kernel 里发" : (mg && c.epOverlap ? "和另一个 micro-batch 的计算重叠（a2a overlap）" : "");
  const cpMeth = r.par.cpMeth;
  const fsU = i => { const k=kind(i); return { d:(k.a==="lin"?fs.pLin:fs.pFull)+(k.f==="dense"?fs.pDen:0), e:k.f==="moe"?fs.pExpL:0 }; };
  const lb = j => j<0 ? "embedding" : j>=m.L ? "输出头" : "第 "+j+" 层";
  const fsX = (i, pass, op, note, mode) => { if(!fs || i<0 || i>=m.L) return; const u=fsU(i);
    for(const [dim,P1,F,lk] of [["fsdp",u.d,fs.fs,fs.lkD],["efsdp",u.e,fs.fe,fs.lkE]])
      for(const x of Par.fsdpDet(P1,F,lk,g,fs.aG,fs.gB)) if(x.pass===pass && x.op===op) X(dim,Object.assign({},x,{what:"第 "+i+" 层"+(dim==="efsdp"?"本卡 expert 的":"的")+" "+x.what+(note||"")}),mode||"bg",i); };
  const tpAttnBlk = k => k.a==="lin" ? "lin" : "attn";

  /* 一层的前向（pass = fwd 或 rc：重算时只走 rc 记录和被重算的算子） */
  const fwdLayer = (i, pass) => { const k=kind(i), rc=pass==="rc", T=p=>rc?p.rc:p.f, pre=rc?"重算 · ":"";
    const tp=P(xsOf("tp",tpAttnBlk(k)),pass);
    X("tp",tp[0],tpM,i,{note:tpNote});
    if(f.kind==="dsa" && k.a==="full") P(xsOf("tpidx","attn"),"fwd").filter(()=>!rc).forEach(x=>X("tp",x,tpM,i));
    if(k.a==="lin"){
      const cl=P(xsOf("cplin","lin"),pass);
      C(pre+"in_proj",T(PC.lpre),"lin");
      if(cl[0] && cl[0].op==="A2A"){ X("cp",cl[0],"block",i); C(pre+"短卷积 + "+m.lin.kind+" chunk kernel + gated norm（"+fN(m.lin.Hv/r.tp/r.cp)+" 个头 × 全序列）",T(PC.lmid),"lin"); X("cp",cl[1],"block",i); }
      else if(cl[0]){ C(pre+"短卷积 + 本段 chunk kernel",T(PC.lmid)*0.5,"lin",{solo:true}); X("cp",cl[0],"block",i,{note:"各段要拿到前面所有段的末状态"}); C(pre+"用前缀状态修正输出 + gated norm",T(PC.lmid)*0.5,"lin"); }
      else C(pre+"短卷积 + "+m.lin.kind+" chunk kernel + gated norm",T(PC.lmid),"lin");
      C(pre+"out_proj",T(PC.lpost),"proj");
    } else {
      const cpx=P(xsOf(k.a==="swa"?"cpswa":"cpkv",k.a==="swa"?"swa":"full"),pass), core=k.a==="swa"?PC.swa:PC.full;
      C(pre+(f.kind==="gqa"?"QKV 投影 + q/k norm + RoPE":f.kind==="dsa"?"q / kv 投影 + indexer 打分":"q / kv 投影（MLA）"),T(PC.pre),"proj");
      if(!cpx.length) C(pre+"attention 核心",T(core),"attn");
      else if(cpMeth==="ulysses"){ X("cp",cpx[0],"block",i); C(pre+"attention 核心（1/CP 的头 × 全序列）",T(core),"attn"); X("cp",cpx[1],"block",i); }
      else if(k.a==="swa"){ X("cp",cpx[0],"block",i); C(pre+"滑窗 attention 核心",T(core),"attn"); }
      else { X("cp",cpx[0],"part",i,{note:cpMeth==="ring"?"每一步算当前块的同时把 K / V 传给下一张卡，只露出第一块":"按 KV 头分块，gather 下一块时算上一块"}); C(pre+"attention 核心（本卡 Q × 全部前缀）",T(core),"attn"); }
      C(pre+"o_proj"+(f.gate?" + 输出门":""),T(PC.post),"proj");
    }
    X("tp",tp[1],tpM,i,{note:tpNote});
    if(k.f==="moe"){
      C(pre+"router + top-k + permute",T(PC.router),"proj");
      P(cnt,pass).forEach(x=>X("ep",x,"block",i,{note:"拿到每个 rank 收多少行才能分配缓冲区"}));
      const ep=P(epAll,pass), et=rc?[]:P(etp,"fwd");
      X("ep",ep[0],epM,i,{note:epNote}); X("etp",et[0],"block",i);
      C(pre+"expert grouped GEMM（fc1 → act → fc2）",T(PC.exp),"moe");
      X("etp",et[1],"block",i); X("ep",ep[1],epM,i,{note:epNote});
      C(pre+"unpermute（k 份加权求和）",T(PC.unp),"proj");
      if(PC.shared.f>0){ const sh=P(xsOf("tp","moe"),pass); X("tp",sh[0],tpM,i); C(pre+"shared expert",T(PC.shared),"proj"); X("tp",sh[1],tpM,i); }
    } else { const dn=P(xsOf("tp","dense"),pass); X("tp",dn[0],tpM,i); C(pre+"dense FFN",T(PC.dense),"proj"); X("tp",dn[1],tpM,i); }
  };
  /* 重算：一整行，计算是这一层被重算的算子，通信是前向里那些调用再来一遍（顺序同前向） */
  const rcLayer = i => { const k=kind(i), blk=tpAttnBlk(k), rc=p=>p.rc;
    const t=(k.a==="lin" ? rc(PC.lpre)+rc(PC.lmid)+rc(PC.lpost) : rc(PC.pre)+rc(k.a==="swa"?PC.swa:PC.full)+rc(PC.post)) + (k.f==="moe" ? rc(PC.router)+rc(PC.exp)+rc(PC.unp)+rc(PC.shared) : rc(PC.dense));
    const cpx=P(k.a==="lin"?xsOf("cplin","lin"):xsOf(k.a==="swa"?"cpswa":"cpkv",k.a==="swa"?"swa":"full"),"rc"), cpBlk=k.a!=="full"||cpMeth==="ulysses";
    const tag = (x, s) => Object.assign({}, x, {what:x.what+"（"+s+"）"});
    const xs=[...P(xsOf("tp",blk),"rc").map(x=>["tp",tag(x,k.a==="lin"?"线性注意力":"注意力"),tpM]), ...cpx.map(x=>["cp",x,cpBlk?"block":"part"]),
      ...(k.f==="moe" ? [...P(cnt,"rc").map(x=>["ep",x,"block"]), ...P(epAll,"rc").map(x=>["ep",x,epM]), ...P(xsOf("tp","moe"),"rc").map(x=>["tp",tag(x,"shared expert"),tpM])] : P(xsOf("tp","dense"),"rc").map(x=>["tp",tag(x,"FFN"),tpM]))];
    if(!(t>0) && !xs.length) return;
    xs.forEach(([dim,x,mode])=>X(dim,x,mode,i,{note:"重算时按前向的顺序再发一遍"}));
    C("重算这一层的前向"+(xs.length?"（通信按前向的顺序再来一遍）":""),t,"oth",{solo:true});
  };
  /* 一层的反向：先重算（如果开了），再 MoE / FFN，再 attention */
  const bwdLayer = i => { const k=kind(i), B=p=>p.b;
    rcLayer(i);
    const tpB = blk => P(xsOf("tp",blk),"bwd");   // Megatron：[AG 输出梯度, AG 再 gather 输入, RS 输入梯度]；torchtitan：[AG, RS]
    const tpBlock = (blk, label, t, cat) => { const x=tpB(blk); X("tp",x[0],tpM,i,{note:tpNote});
      if(mg && x.length===3){ X("tp",x[1],"bg",i,{note:"和 dgrad GEMM 重叠"}); X("tp",x[2],"bg",i,{note:"和 wgrad GEMM 重叠"}); C(label,t,cat); }
      else { C(label,t,cat); X("tp",x[1],tpM,i,{note:tpNote}); } };
    if(k.f==="moe"){
      if(PC.shared.f>0) tpBlock("moe","shared expert 反向",B(PC.shared),"proj");
      const ep=P(epAll,"bwd"), et=P(etp,"bwd");
      X("ep",ep[0],epM,i,{note:epNote}); X("etp",et[0],"block",i);
      C("expert dgrad + wgrad",B(PC.exp),"moe");
      X("etp",et[1],"block",i); X("ep",ep[1],epM,i,{note:epNote});
      C("unpermute / router / permute 反向",B(PC.unp)+B(PC.router),"proj");
    } else tpBlock("dense","dense FFN 反向",B(PC.dense),"proj");
    const blk=tpAttnBlk(k), tx=tpB(blk);
    X("tp",tx[0],tpM,i,{note:tpNote});
    if(k.a==="lin"){
      C("out_proj 反向",B(PC.lpost),"proj");
      const cl=P(xsOf("cplin","lin"),"bwd");
      if(cl[0] && cl[0].op==="A2A"){ X("cp",cl[0],"block",i); C("gated norm + chunk kernel + 短卷积 反向",B(PC.lmid),"lin"); X("cp",cl[1],"block",i); }
      else if(cl[0]){ C("chunk kernel 反向（本段）",B(PC.lmid)*0.5,"lin",{solo:true}); X("cp",cl[0],"block",i); C("用状态梯度修正 + 短卷积反向",B(PC.lmid)*0.5,"lin"); }
      else C("gated norm + chunk kernel + 短卷积 反向",B(PC.lmid),"lin");
      if(mg && tx.length===3){ X("tp",tx[1],"bg",i,{note:"和 dgrad GEMM 重叠"}); X("tp",tx[2],"bg",i,{note:"和 wgrad GEMM 重叠"}); C("in_proj 反向",B(PC.lpre),"proj"); }
      else { C("in_proj 反向",B(PC.lpre),"proj"); X("tp",tx[1],tpM,i); }
    } else {
      C("o_proj 反向",B(PC.post),"proj");
      const cpx=P(xsOf(k.a==="swa"?"cpswa":"cpkv",k.a==="swa"?"swa":"full"),"bwd"), core=k.a==="swa"?PC.swa:PC.full;
      if(!cpx.length) C("attention 反向",B(core),"attn");
      else if(cpMeth==="ulysses"){ X("cp",cpx[0],"block",i); C("attention 反向",B(core),"attn"); X("cp",cpx[1],"block",i); }
      else if(k.a==="swa"){ X("cp",cpx[0],"block",i); C("滑窗 attention 反向",B(core),"attn"); X("cp",cpx[1],"block",i); }
      else if(cpMeth==="ring"){ X("cp",cpx[0],"part",i,{note:"和分块反向交错"}); X("cp",cpx[1],"part",i,{note:"dK / dV 跟着块走，最后一步回到本卡"}); C("attention 反向（分块）",B(core),"attn"); }
      else { X("cp",cpx[0],"part",i,{note:"按 KV 头分块"}); C("attention 反向",B(core),"attn"); X("cp",cpx[1],"block",i,{note:"QKV 投影的反向要等本卡 token 的 dK / dV"}); }
      if(f.kind==="dsa" && k.a==="full") P(xsOf("tpidx","attn"),"bwd").forEach(x=>X("tp",x,tpM,i));
      if(mg && tx.length===3){ X("tp",tx[1],"bg",i,{note:"和 dgrad GEMM 重叠"}); X("tp",tx[2],"bg",i,{note:"和 wgrad GEMM 重叠"}); C("QKV 投影反向",B(PC.pre),"proj"); }
      else { C("QKV 投影反向",B(PC.pre),"proj"); X("tp",tx[1],tpM,i); }
    }
  };

  const i0=L[0], i1=L[L.length-1];
  if(show!=="bwd"){
    if(fs){ fsX(i0,"fwd","AG","（上一层计算时预取）"); C(lb(i0-1)+" 前向（不展开）",0,"oth",{ghost:true}); }
    for(const i of L){ sec("第 "+i+" 层 · "+esc(name(kind(i)))+" · <span>前向</span>");
      if(fs) fsX(i+1,"fwd","AG","（预取，下一层用）");
      fwdLayer(i,"fwd"); }
  }
  if(show!=="fwd"){
    if(fs){ if(fs.aG===2) fsX(i1,"bwd","AG","（上一层反向时预取）"); C(lb(i1+1)+" 反向（不展开）",0,"oth",{ghost:true}); }
    for(const i of L.slice().reverse()){ sec("第 "+i+" 层 · "+esc(name(kind(i)))+" · <span>反向"+(r.ops.some(o=>o.rc)?"（先重算这一层的前向）":"")+"</span>");
      if(fs && fs.aG===2) fsX(i-1,"bwd","AG","（预取，下一层反向用）");
      bwdLayer(i);
      if(fs) fsX(i,"bwd","RS","（这一层反向算完就发，和下一层反向重叠）"); }
    if(fs) C(lb(i0-1)+" 反向（不展开）",0,"oth",{ghost:true});
  }
  /* 每个 micro-batch / 每个 step 各一次 */
  const ppx=xsOf("pp","stage");
  if(ppx.length || r.stepX.length || fs){
    sec("每个 micro-batch、每个 step 各一次",{sub:true});
    if(ppx.length){ X("pp",ppx[0],"bg","mb",{note:"1F1B 里和别的 micro-batch 的计算重叠"}); C("本 stage 最后一层前向算完",0,"oth",{ghost:true});
      X("pp",ppx[1],"bg","mb",{note:"1F1B 里和别的 micro-batch 的计算重叠"}); C("本 stage 第一层反向算完",0,"oth",{ghost:true}); }
    if(mg){ const rs=r.stepX.filter(x=>x.op==="RS"), ag=r.stepX.filter(x=>x.op==="AG");
      rs.forEach(x=>X(x.dim,x,c.overlapGrad?"bg":"block","step",{note:c.overlapGrad?"按桶在最后一个 micro-batch 的反向里发（overlap_grad_reduce）":""}));
      if(c.overlapGrad && rs.length) C("最后一个 micro-batch 的反向",0,"oth",{ghost:true});
      C("optimizer："+r.optNote+"（每卡只更新自己那 1/(DP×CP) 份）",r.tOpt,"oth",{solo:true});
      ag.forEach(x=>X(x.dim,x,c.overlapParam?"bg":"block","step",{note:c.overlapParam?"按桶在下一步第一个 micro-batch 的前向里发（overlap_param_gather）":""}));
      if(c.overlapParam && ag.length) C("下一步第一个 micro-batch 的前向",0,"oth",{ghost:true}); }
    else { r.stepX.forEach(x=>X(x.dim,x,"block","step",{note:"每步一次，和最后的反向部分重叠"}));
      C("optimizer："+r.optNote+"（FSDP：每卡只更新自己那 1/F 份；梯度已经在每层 reduce-scatter 掉了）",r.tOpt,"oth",{solo:true}); }
  }
  return ev;
}
function trnGroups(r){
  const {topo}=ctx(), fs=r.par.fsdp, mg=S.trn.fw==="megatron", N=Math.round(r.dp*r.tp*r.cp*r.pp), Nst=N/r.pp, kd=l=>l?(l.n<=1?"":l.intra===false?"IB":"NVLink"):"NVLink";
  const G = (n, stride, lk) => ({ n, lk:n>1?kd(lk):"", mem:n>1?xMembers(n,stride):"" });
  const o = { tp:G(r.tp,1,r.lk.tp), cp:G(r.cp,r.tp,r.lk.cp), ep:G(r.ep,mg?r.etp:1,r.lk.ep), etp:G(r.etp,1,r.lk.tp), pp:G(r.pp,Nst,r.lk.pp) };
  if(fs){ o.fsdp=G(fs.fs,r.tp,fs.lkD); o.efsdp=G(fs.fe,r.ep,fs.lkE); o.rep=G(fs.R,Nst/fs.R,fs.lkR); }
  for(const x of r.stepX) if(!o[x.dim]) o[x.dim]=G(x.n,x.dim==="dp"?r.tp:r.ep*r.etp,x.lk);
  return o;
}
function renderTrnComm(){
  const {m,hw,topo}=ctx(), c=S.trn, r=trnRun(), mg=c.fw==="megatron", fs=r.par.fsdp, {kind,name}=xKinds(m);
  const L=xSample(m,XV.trn.n), ev=trnPlan(r,L,XV.trn.show), groups=trnGroups(r);
  xSeg("x-t-n",[[1,"1 层"],[2,"2 层"],[3,"3 层"]],XV.trn.n,v=>{ XV.trn.n=+v; renderTrnComm(); });
  xSeg("x-t-show",[["both","前向 + 反向"],["fwd","只看前向"],["bwd","只看反向"]],XV.trn.show,v=>{ XV.trn.show=v; renderTrnComm(); });
  const lanes=xGrid("x-t-grid",ev,groups);
  $("x-t-lede").innerHTML=(mg?"Megatron-Core":"torchtitan")+" · "+esc(m.short)+" · "+hw.name+" · "+esc(topo.name)+"。TP "+r.tp+" / CP "+r.cp+" / EP "+r.ep+(r.etp>1?" / ETP "+r.etp:"")+(r.pp>1?" / PP "+r.pp:"")+" / "+(fs?"FSDP "+fs.fs+(fs.fe>1?"（expert "+fs.fe+"）":"")+(fs.R>1?" × HSDP "+fs.R:""):"DP "+r.dp+"（分布式优化器）")+
    "，每卡每个 micro-batch "+V(fN(r.Tc))+" token，重算："+esc({full:"整层",selective:"选择性",default:"默认（只重算 w13）",none:"不重算"}[c.recompute]||c.recompute)+"。这一张跟着左侧的框架走（和“"+'<a href="#train/step">一个 step</a>'+"”同一个 <code>Train.run</code>）；下面几节把 Megatron 换成 torchtitan 的 FSDP 来讲。";
  /* 样例层的字节 */
  const lay = i => ev.filter(e=>e.k==="x"&&e.lay===i);
  xVolume("x-t-vol",L.map(i=>({label:"第 "+i+" 层 <span>"+esc(name(kind(i)))+"</span>",parts:lanes.map(dim=>({dim,v:sum(lay(i).filter(e=>e.dim===dim),e=>e.bytes)}))})));
  /* 按维度汇总：样例层、每 micro-batch、每 step */
  const byMB={}, add=(o,k,b,t)=>{ o[k]=o[k]||{b:0,t:0}; o[k].b+=b; o[k].t+=t; };
  const dimOf = x => x.dim || ({tp:"tp",tpidx:"tp",cpkv:"cp",cpswa:"cp",cplin:"cp",a2a:"ep",cnt:"ep",etp:"etp",pp:"pp"})[x.key];
  for(const x of r.comm){ const dm=dimOf(x); if(!dm) continue; const k=x.perStage?1:1/r.pp; add(byMB,dm,x.bytes*k,x.t*k); }
  if(fs){ add(byMB,"fsdp",fs.bDense,fs.tDense); add(byMB,"efsdp",fs.bExp,fs.tExp); }
  const bySt={}; for(const k in byMB) add(bySt,k,byMB[k].b*r.microsPerStep,byMB[k].t*r.microsPerStep); for(const x of r.stepX) add(bySt,x.dim,x.bytes,x.t);
  const cnt = (dim, i) => { const xs=lay(i).filter(e=>e.dim===dim), n=p=>xs.filter(e=>e.pass===p).length; return xs.length ? [["fwd","前向"],["rc","重算"],["bwd","反向"]].filter(([p])=>n(p)).map(([p,l])=>l+" "+n(p)).join(" · ") : "—"; };
  const dims=XORDER.filter(k=>lanes.includes(k)||bySt[k]);
  const stTot=sum(dims,k=>bySt[k]?bySt[k].b:0);
  $("x-t-tbl").innerHTML='<thead><tr><th>维度</th><th>组（GPU 0 所在）</th>'+L.map(i=>'<th>第 '+i+' 层：次数 / 字节</th>').join("")+'<th>每 micro-batch</th><th>每 step（'+r.microsPerStep+' 个 micro-batch）</th><th>每 step 线上时间</th></tr></thead><tbody>'+
    dims.map(k=>{ const G=groups[k]||{}, s=bySt[k]||{b:0,t:0}, mb=byMB[k];
      return '<tr><td><span class="sw" style="background:'+css(XD[k][1])+'"></span>'+esc(XD[k][0])+'</td><td>'+(G.n||"—")+' 卡'+(G.lk?' · '+G.lk:'')+(G.mem?'<br><span class="tag">'+esc(G.mem)+'</span>':'')+'</td>'+
        L.map(i=>{ const b=sum(lay(i).filter(e=>e.dim===k),e=>e.bytes); return '<td class="num">'+(b?cnt(k,i)+'<br>'+fB(b):'—')+'</td>'; }).join("")+
        '<td class="num">'+(mb?fB(mb.b):"—")+'</td><td class="num">'+fB(s.b)+' <span class="tag">'+pc(s.b/stTot)+'</span></td><td class="num">'+fT(s.t)+'</td></tr>'; }).join("")+
    '<tr class="hl"><td>合计</td><td></td>'+L.map(i=>'<td class="num">'+fB(sum(lay(i),e=>e.bytes))+'</td>').join("")+'<td class="num">'+fB(sum(dims,k=>byMB[k]?byMB[k].b:0))+'</td><td class="num">'+fB(stTot)+'</td><td class="num">'+fT(sum(dims,k=>bySt[k]?bySt[k].t:0))+'</td></tr></tbody>';
  const big=dims.slice().sort((a,b)=>(bySt[b]?bySt[b].b:0)-(bySt[a]?bySt[a].b:0))[0];
  $("x-t-note").innerHTML="每卡一个 step 在线上发出 "+V(fB(stTot))+"，最多的是 "+esc(XD[big]?XD[big][0]:"—")+"（"+pc((bySt[big]?bySt[big].b:0)/stTot)+"）。"+
    "层内的字节随 token 数变（TP、CP、EP 搬的是激活），FSDP 和 DP 的字节只随参数量变；这就是为什么短序列偏向少切激活、多切参数，长序列反过来。"+
    (r.pp>1?"表里每 micro-batch 的数是一个 PP stage 的（约 "+Math.round(m.L/r.pp)+" 层）。":"");
}

/* ================= 推理：当前引擎、当前阶段（和“一步 decode / prefill”同一个 Infer） ================= */
function infPlan(r, L){
  const {m}=ctx(), c=S.inf, {kind,name}=xKinds(m), dec=c.phase==="decode";
  const per = pred => sum(r.ops.filter(pred),o=>o.t/o.layers);
  const cores=r.ops.filter(o=>o.key==="acore");
  const PI = { pre:per(o=>o.group==="attn"&&!["acore","agate","auv","aout"].includes(o.key)), full:cores[0]?cores[0].t/cores[0].layers:0, swa:cores[1]?cores[1].t/cores[1].layers:0, post:per(o=>["agate","auv","aout"].includes(o.key)),
    lin:per(o=>o.group==="lin"), router:per(o=>["router","latd","moeq"].includes(o.key)), exp:per(o=>o.key==="moe"), latu:per(o=>o.key==="latu"), shared:per(o=>o.key==="shared"), dense:per(o=>o.key==="dense"),
    lm:per(o=>o.key==="lm"), samp:per(o=>o.key==="samp") };
  const one = key => { const e=r.comm.find(x=>x.key===key); return e ? { op:{ar:"AR",latar:"AR",dpg:"AG",dps:"RS",disp:"A2A",comb:"A2A",cnt:"sync",lmg:"AG"}[key], bytes:e.bytes/e.layers, t:e.t/e.layers, ov:e.overlap } : null; };
  const ev=[], C=(label,t,cat,o)=>ev.push(Object.assign({k:"c",label,t,cat},o||{})), sec=(label,o)=>ev.push(Object.assign({k:"s",label},o||{}));
  const X=(dim,key,what,lay,o)=>{ const x=one(key); if(!x) return; const mode=x.ov==="moe"?"fused":x.ov==="tbo"?"part":"block";
    ev.push(Object.assign({k:"x",dim,mode,lay,pass:"fwd",what,note:x.ov==="tbo"?"两个 micro-batch 交替（TBO）：一个算的时候另一个在通信":x.ov==="moe"?"融合在 MoE kernel 里":""},x,o||{})); };
  const ag = r.moePar==="ag", epDim = ag ? "ep" : "dpa";
  const arWhat = w => w+(r.fused?" + 残差 + RMSNorm（融合）":"");
  for(const i of L){ const k=kind(i);
    sec("第 "+i+" 层 · "+esc(name(k))+" · <span>"+(dec?"一步 decode":"一个 prefill chunk")+"</span>");
    if(k.a==="lin") C(m.lin.kind+"（in_proj → 状态更新 → out_proj）",PI.lin,"lin");
    else { C("QKV 投影"+(m.full.kind!=="gqa"?"（MLA 下投影 + 上投影）":""),PI.pre,"proj"); C((k.a==="swa"?"滑窗 ":"")+"attention 核心",k.a==="swa"?PI.swa:PI.full,"attn"); C("o_proj"+(m.full.gate?" + 输出门":""),PI.post,"proj"); }
    X("tp","ar",arWhat("attention 输出"),i);
    if(k.f==="moe"){
      X(epDim,"dpg",ag?"EP：所有 rank 的 token 收齐":"DP-attention：各 rank 的 token 收齐进 MoE",i);
      C("router + top-k"+(m.moe.Dlat?" + latent 下投影":""),PI.router,"proj");
      if(PI.shared>0) C("shared expert（不等 dispatch）",PI.shared,"proj");
      X("ep","cnt","token 计数交换 + D2H 同步",i); X("ep","disp","dispatch：token 发到 expert 所在卡",i);
      C("routed experts",PI.exp,"moe");
      X("ep","comb","combine：expert 输出发回",i);
      X("tp","latar","latent 宽度的部分和（"+m.moe.Dlat+" 维）",i);
      if(PI.latu>0) C("latent 上投影",PI.latu,"proj");
      X(epDim,"dps",ag?"EP：输出 reduce-scatter 回各 rank":"DP-attention：输出切回各 rank",i);
      X("tp","ar",arWhat("MoE 输出"),i);
    } else { C("dense FFN",PI.dense,"proj"); X("tp","ar",arWhat("FFN 输出"),i); }
  }
  sec("每步一次",{sub:true});
  C("lm_head",PI.lm,"lm"); X("tp","lmg","logits（词表并行）","step"); C("采样",PI.samp,"lm");
  return ev;
}
/* 推理：默认看左侧的当前配置；它一张卡一个实例（没有通信）时，换成这个拓扑上第一个多卡的部署预设 */
function infCommCfg(){
  const cfgs=toConfigs(S.topo), cur=infRun();
  const opts=[["cur","左侧当前配置（"+ENGN[S.inf.engine]+"，"+S.inf.N+" 卡）"]].concat(cfgs.filter(cf=>cf.cfg.N>1).map(cf=>[cf.id,cf.eng+" · "+cf.label]));
  let id=XV.inf.cfg; if(!opts.some(o=>o[0]===id)) id="cur";
  let auto=false; if(id==="cur" && !cur.comm.length && opts.length>1 && !XV.inf.pinned){ id=opts[1][0]; auto=true; }
  const sel=$("x-i-cfg"); sel.innerHTML=opts.map(o=>'<option value="'+esc(o[0])+'">'+esc(o[1])+'</option>').join(""); sel.value=id;
  sel.onchange=e=>{ XV.inf.cfg=e.target.value; XV.inf.pinned=true; renderInfComm(); };
  const cf=cfgs.find(x=>x.id===id);
  return { over: cf ? Object.assign({},cf.cfg,{engine:cf.cfg.engine||cf.engine}) : {}, cf, auto };
}
function renderInfComm(){
  const {m,hw,topo}=ctx(), pick=infCommCfg(), c=Object.assign({},S.inf,pick.over), r=infRun(pick.over), {kind,name}=xKinds(m), dec=c.phase==="decode";
  const L=xSample(m,XV.inf.n), ev=infPlan(r,L), tpA=Math.round(r.tpA), lk=n=>n>1?"NVLink":"";
  const groups={ tp:{n:tpA,lk:lk(tpA),mem:tpA>1?xMembers(tpA,1):""}, dpa:{n:r.dpA,lk:lk(r.dpA),mem:r.dpA>1?xMembers(r.dpA,tpA):""}, ep:{n:r.N,lk:lk(r.N),mem:r.N>1?xMembers(r.N,1):""} };
  xSeg("x-i-n",[[1,"1 层"],[2,"2 层"],[3,"3 层"]],XV.inf.n,v=>{ XV.inf.n=+v; renderInfComm(); });
  const lanes=xGrid("x-i-grid",ev,groups);
  $("x-i-lede").innerHTML=(pick.auto?"左侧当前配置是一张卡一个实例，一层里没有集合通信，所以这里先换成本拓扑上第一个多卡的部署预设（下拉框里可以换）。":"")+ENGN[c.engine]+" · "+esc(m.short)+" · "+hw.name+"。一个实例 "+c.N+" 卡：attention TP "+tpA+" × DP "+r.dpA+"，MoE "+esc(MOENAME[r.moePar]||r.moePar)+"；"+(dec?"每个 DP rank "+r.Breq+" 个请求、一步 "+fN(r.rowsG)+" 行（含补齐"+(r.specK?"和投机的 "+(r.specK+1)+" 个位置":"")+"）":"一个 "+fN(c.C)+" token 的 prefill chunk")+
    "。推理只有前向：TP 每层两次 all-reduce（attention 后、FFN 后），EP 每层一次 dispatch 一次 combine，消息大小随“行数 × D”走。跟着左侧的推理参数走（"+'<a href="#infer/step">一步 decode / prefill</a>'+" 同一个模型）。";
  const lay = i => ev.filter(e=>e.k==="x"&&e.lay===i);
  xVolume("x-i-vol",L.map(i=>({label:"第 "+i+" 层 <span>"+esc(name(kind(i)))+"</span>",parts:lanes.map(dim=>({dim,v:sum(lay(i).filter(e=>e.dim===dim),e=>e.bytes)}))})));
  const dimOf = key => ({ar:"tp",latar:"tp",lmg:"tp",disp:"ep",comb:"ep",cnt:"ep",dpg:r.moePar==="ag"?"ep":"dpa",dps:r.moePar==="ag"?"ep":"dpa"})[key];
  const bySt={}; for(const x of r.comm){ const k=dimOf(x.key); if(!k) continue; bySt[k]=bySt[k]||{b:0,t:0}; bySt[k].b+=x.bytes; bySt[k].t+=x.t; }
  const dims=XORDER.filter(k=>bySt[k]), tot=sum(dims,k=>bySt[k].b), toks = dec ? r.Breq*(1+(r.specK?c.accept:0)) : c.C;
  $("x-i-tbl").innerHTML='<thead><tr><th>维度</th><th>组（GPU 0 所在）</th>'+L.map(i=>'<th>第 '+i+' 层：次数 / 字节</th>').join("")+'<th>一步（'+m.L+' 层 + 输出头）</th><th>一步线上时间</th><th>每个'+(dec?"输出":"输入")+' token</th></tr></thead><tbody>'+
    dims.map(k=>{ const G=groups[k]||{}, s=bySt[k];
      return '<tr><td><span class="sw" style="background:'+css(XD[k][1])+'"></span>'+esc(XD[k][0])+'</td><td>'+(G.n||"—")+' 卡'+(G.lk?' · '+G.lk:'')+(G.mem?'<br><span class="tag">'+esc(G.mem)+'</span>':'')+'</td>'+
        L.map(i=>{ const xs=lay(i).filter(e=>e.dim===k); return '<td class="num">'+(xs.length?xs.length+' 次<br>'+fB(sum(xs,e=>e.bytes)):'—')+'</td>'; }).join("")+
        '<td class="num">'+fB(s.b)+' <span class="tag">'+pc(s.b/tot)+'</span></td><td class="num">'+fT(s.t)+'</td><td class="num">'+fB(s.b/Math.max(1,toks))+'</td></tr>'; }).join("")+
    (dims.length?'<tr class="hl"><td>合计</td><td></td>'+L.map(i=>'<td class="num">'+fB(sum(lay(i),e=>e.bytes))+'</td>').join("")+'<td class="num">'+fB(tot)+'</td><td class="num">'+fT(sum(dims,k=>bySt[k].t))+'</td><td class="num">'+fB(tot/Math.max(1,toks))+'</td></tr>':'<tr><td colspan="'+(5+L.length)+'">这个配置一个实例只有一张卡（或者各卡独立），没有层内通信。</td></tr>')+'</tbody>';
  $("x-i-note").innerHTML=!dims.length ? "TP1 × 多实例：每张卡是一个独立副本，一步里没有任何集合通信，代价是每张卡都要放下并读完整份权重。" :
    "一步每卡在线上发出 "+V(fB(tot))+"，线上时间 "+V(fT(sum(dims,k=>bySt[k].t)))+"（一步 "+fT(r.stepT)+"）。decode 的消息很小（几十 KB 到几 MB），时间主要是每次调用的固定延迟，所以 AR 融合、one-shot / two-shot 和 DeepEP low-latency 这类省延迟的办法比省字节更要紧；prefill 的消息大几百倍，才开始受带宽限制。";
}
