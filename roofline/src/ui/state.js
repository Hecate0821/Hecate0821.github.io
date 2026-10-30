/* ---------- 状态 ---------- */
const S = { modelKey:"qwen35", model:clone(MODELS.qwen35), hw:"H200", topo:"hgx8", glob:clone(GLOB),
  inf:clone(BASE_INF), trn:clone(BASE_TRN),
  infPreset:"rec", trnPreset:"slime", tab:"overview", sub:"",
  q:{ hw:"B200", scheme:"nvfp4", fuse:"fused", shape:"up", N:1024, K:2048, M:64, T:512, ep:8, align:"auto", recipe:"nvfp4hp", Mt:8192, nMicro:4, tokFused:false, roll:0.7, deqRate:40, BM:128 } };
function getP(path){ return path.split(".").reduce((o,k)=>o==null?o:o[k], S); }
function setP(path,v){ const ks=path.split("."); let o=S; for(let i=0;i<ks.length-1;i++) o=o[ks[i]]; o[ks[ks.length-1]]=v; }

function setTopo(t, hw){
  S.topo=t; if(hw && hwOK(hw,t)) S.hw=hw; else if(!hwOK(S.hw,t)) S.hw=prefHW(S.modelKey,t);
  $("g-topo").value=S.topo; $("g-hw").value=S.hw; }
const needOK = p => !p.need || (TOPO[S.topo].gpus>=TOPO[p.need.topo].gpus && (!p.need.hw || S.hw===p.need.hw));
const firstOK = P => (P.find(p=>(p.cfg||p.rl) && needOK(p)) || P[0]).id;
function applyNeed(p, sw){ if(sw!==false && !needOK(p)) setTopo(p.need.topo, p.need.hw); }
function applyInf(id){
  const P=infPresets(S.inf.engine,S.modelKey,S.topo), p=P.find(x=>x.id===id)||P[0];
  S.infPreset=p.id; if(!p.cfg) return;
  const keep={engine:S.inf.engine, phase:S.inf.phase, n:S.inf.n, P:S.inf.P, C:S.inf.C, bi:S.inf.bi, biFactor:S.inf.biFactor, implEff:S.inf.implEff};
  S.inf=Object.assign(clone(BASE_INF),keep,p.cfg);
}
function applyTrn(id, sw){
  const P=trnPresets(S.trn.fw,S.modelKey,S.topo), p=P.find(x=>x.id===id)||P[0];
  S.trnPreset=p.id; if(!p.cfg) return; applyNeed(p, sw);
  S.trn=Object.assign(clone(BASE_TRN),{fw:S.trn.fw, S:S.trn.S, Gtok:S.trn.Gtok}, S.trn.fw==="torchtitan"?{cpType:"allgather",linCP:"state",dispatcher:"standard",recompute:"default",offload:false}:{}, p.cfg);
}
function applyAll(){ applyInf("rec"); applyTrn(firstOK(trnPresets(S.trn.fw,S.modelKey,S.topo)),false); }
/* ---------- 全局选择器 ---------- */
function fillSel(sel, obj, val, label){ sel.innerHTML=Object.keys(obj).map(k=>'<option value="'+k+'">'+esc(label?label(obj[k]):obj[k].name)+'</option>').join(""); sel.value=val; }
function fillOpts(sel, list, val){ sel.innerHTML=list.map(x=>'<option value="'+x[0]+'">'+esc(x[1])+'</option>').join(""); sel.value=val; }
fillSel($("g-model"),MODELS,S.modelKey); fillSel($("g-hw"),HW,S.hw,h=>h.name+" · "+h.mem+" GB"); fillSel($("g-topo"),TOPO,S.topo);
$("g-model").addEventListener("change",e=>{ S.modelKey=e.target.value; S.model=clone(MODELS[S.modelKey]); setTopo(S.topo, prefHW(S.modelKey,S.topo)); applyAll(); qShape(); buildEditor(); syncInputs(); render(); });
$("g-hw").addEventListener("change",e=>{ if(!hwOK(e.target.value,S.topo)){ setTopo(e.target.value.startsWith("GB")?"nvl64":"hgx8", e.target.value); } else S.hw=e.target.value;
  if(S.inf.engine==="titan" && S.infPreset!=="custom") applyInf(S.infPreset); syncInputs(); render(); });
$("g-topo").addEventListener("change",e=>{ setTopo(e.target.value);
  applyInf(S.infPreset==="custom"?"rec":S.infPreset);
  const tp=trnPresets(S.trn.fw,S.modelKey,S.topo), cur=tp.find(x=>x.id===S.trnPreset&&x.cfg); applyTrn(cur&&needOK(cur)?cur.id:firstOK(tp),false);
  syncInputs(); render(); });
$("i-preset").addEventListener("change",e=>{ applyInf(e.target.value); syncInputs(); render(); });
$("t-preset").addEventListener("change",e=>{ applyTrn(e.target.value); syncInputs(); render(); });
$("model-reset").addEventListener("click",()=>{ S.model=clone(MODELS[S.modelKey]); buildEditor(); render(); });

/* ---------- 双向绑定 ---------- */
const LIVE_INF = /^inf\.(phase|B|n|P|C|bi|biFactor|implEff)$/, LIVE_TRN=/^trn\.(tok|S|Gtok)$/;
function syncInputs(){
  qa("[data-bind]").forEach(el=>{ const v=getP(el.dataset.bind); if(v===undefined) return; if(el.type==="checkbox") el.checked=!!v; else el.value=String(v); });
  qa("[data-out]").forEach(el=>{ el.textContent=fInt(getP(el.dataset.out)); });
}
function onBind(el){
  const path=el.dataset.bind; let v = el.type==="checkbox" ? el.checked : el.value;
  if(el.type==="number"||el.type==="range") v=parseFloat(v);
  if(typeof v==="number" && !isFinite(v)) return;
  setP(path,v);
  if(path==="inf.engine"){ applyInf("rec"); syncInputs(); }
  else if(path.startsWith("inf.") && !LIVE_INF.test(path)) S.infPreset="custom";
  if(path==="trn.fw"){ applyTrn(firstOK(trnPresets(v,S.modelKey,S.topo)),false); syncInputs(); }
  else if(path.startsWith("trn.") && !LIVE_TRN.test(path)) S.trnPreset="custom";
  if(path==="model.full.kind"||path==="model.lin.kind") buildEditor();
  if(path==="q.shape"){ qShape(); syncInputs(); } else if(path==="q.N"||path==="q.K"){ S.q.shape="custom"; syncInputs(); }
  if(path.startsWith("model.")) qShape();
  qa('[data-out="'+path+'"]').forEach(o=>o.textContent=fInt(v));
  render();
}
document.addEventListener("input",e=>{ const el=e.target; if(el.dataset && el.dataset.bind && el.tagName!=="SELECT" && el.type!=="checkbox") onBind(el); });
document.addEventListener("change",e=>{ const el=e.target; if(el.dataset && el.dataset.bind && (el.tagName==="SELECT"||el.type==="checkbox")) onBind(el);
  if(el.dataset && el.dataset.rcmod){ const s=new Set(S.trn.rcModules); el.checked?s.add(el.dataset.rcmod):s.delete(el.dataset.rcmod); S.trn.rcModules=[...s]; S.trnPreset="custom"; render(); } });

/* ---------- 模型编辑器 ---------- */
function buildEditor(){
  const m=S.model, k=m.full.kind, fk=m.full;
  if(k!=="gqa"){ for(const [a,b] of [["H",64],["qLora",1536],["kvLora",512],["nope",128],["rope",64],["vh",128]]) if(fk[a]==null) fk[a]=b; }
  else { for(const [a,b] of [["Hq",32],["Hkv",4],["dh",128]]) if(fk[a]==null) fk[a]=b; }
  if(k==="dsa"){ for(const [a,b] of [["topk",2048],["idxH",32],["idxD",128],["kpool",4]]) if(fk[a]==null) fk[a]=b; }
  const F=[["基本"],["D","hidden D"],["L","层数 L"],["V","词表 V"],
    ["全注意力层"],["full.n","层数"],["full.kind","类型","selA"],
    ...(k==="gqa"?[["full.Hq","Q 头数"],["full.Hkv","KV 头数"],["full.dh","head_dim"]]
       :[["full.H","头数"],["full.qLora","q_lora_rank"],["full.kvLora","kv_lora_rank"],["full.nope","qk_nope_dim"],["full.rope","qk_rope_dim"],["full.vh","v_head_dim"]]),
    ...(k==="dsa"?[["full.topk","DSA top-k"],["full.idxH","indexer 头数"],["full.idxD","indexer 维度"],["full.kpool","KPool 池大小"]]:[]),
    ["线性注意力层"],["lin.n","层数"],["lin.kind","类型","selL"],["lin.Hk","QK 头数"],["lin.Hv","V 头数"],["lin.dk","key 维度"],["lin.dv","value 维度"],
    ["MoE"],["moe.E","routed expert 数 E"],["moe.k","top-k"],["moe.F","moe_intermediate F"],["moe.Fs","shared expert 合计 intermediate"],["moe.Dlat","latent 维度（0 = 不用）"],
    ["dense.n","前几层 dense FFN"],["dense.F","dense intermediate"]];
  let h="";
  for(const f of F){
    if(f.length===1){ h+='<h3>'+esc(f[0])+'</h3>'; continue; }
    if(f[2]==="selA") h+='<label class="ctl"><span>'+f[1]+'</span><select data-bind="model.'+f[0]+'"><option value="gqa">GQA</option><option value="mla">MLA</option><option value="dsa">DSA（MLA + 稀疏）</option></select></label>';
    else if(f[2]==="selL") h+='<label class="ctl"><span>'+f[1]+'</span><select data-bind="model.'+f[0]+'"><option value="GDN">Gated DeltaNet</option><option value="KDA">KDA</option></select></label>';
    else h+='<label class="ctl"><span>'+esc(f[1])+'</span><input type="number" min="0" step="1" data-bind="model.'+f[0]+'"></label>';
  }
  $("model-editor").innerHTML=h; syncInputs();
}

/* ---------- 共用 ---------- */
const ctx = () => ({ m:S.model, d:derive(S.model), hw:HW[S.hw], topo:TOPO[S.topo], g:S.glob });
function kpis(el, list){ el.innerHTML=list.map(k=>'<div class="kpi'+(k[3]?' bad':'')+'"><span class="k">'+k[0]+'</span><span class="n">'+k[1]+'</span><span class="f">'+esc(k[2]||"")+'</span></div>').join(""); }
function presetSel(sel, srcEl, P, cur){ sel.innerHTML=P.map(p=>'<option value="'+p.id+'">'+esc(p.label+(needOK(p)?"":"（需 "+TOPO[p.need.topo].gpus+" 卡"+(p.need.hw?" "+p.need.hw:"")+"，选中会切换拓扑）"))+'</option>').join(""); sel.value=cur; const p=P.find(x=>x.id===cur); srcEl.textContent=p?p.src:""; }
function setWarn(list){ const gw=$("g-warn"); gw.hidden=!list.length; gw.innerHTML=list.map(esc).join("<br>"); }
const ul = items => '<ul class="tight">'+items.filter(Boolean).map(x=>'<li>'+x+'</li>').join("")+'</ul>';
const V = s => '<span class="v">'+s+'</span>';
function table(el, head, rows, numCols){ const nc=new Set(numCols||[]);
  el.innerHTML='<thead><tr>'+head.map((h,i)=>'<th'+(nc.has(i)?' style="text-align:right"':'')+'>'+h+'</th>').join("")+'</tr></thead><tbody>'+
    rows.map(r=>'<tr'+(r.cls?' class="'+r.cls+'"':'')+'>'+(r.cells||r).map((c,i)=>'<td'+(nc.has(i)?' class="num"':'')+'>'+c+'</td>').join("")+'</tr>').join("")+'</tbody>'; }
const sw = c => '<span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;background:'+c+'"></span>';
function roofSeries(hw,g,peaks,x0,x1){ const bw=hw.hbm*1e9*g.etaM, xs=logspace(x0,x1,200);
  return peaks.map((p,i)=>({pts:xs.map(I=>[I,Math.min(p.v*g.etaC,bw*I)]),color:css("ink"),w:i?1.2:1.8,dash:i?"6 4":null})); }
