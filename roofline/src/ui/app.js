/* ---------- 页面与渲染：hash = 页 / 子页 / 锚点 ---------- */
const NAV = { overview:[], train:["step","par","quant","calib"], infer:["step","deploy","quant","spec"], rl:["mismatch","lowprec","qat"], appendix:["data","quant","models","method"] };
const OLD = { models:"appendix/models", par:"train/par", "par/calib":"train/calib", quant:"infer/quant", public:"appendix/data", method:"appendix/method" };   // 旧的七个 tab
const VIEW = {
  "overview": renderOverview,
  "train/step": renderTrain, "train/par": ()=>{ syncTrnRack(); renderPar(); }, "train/quant": renderQuantTrain, "train/calib": renderCalib,
  "infer/step": renderInfer, "infer/deploy": ()=>{ syncInfRack(); renderInferTO(); }, "infer/quant": renderQuantInfer, "infer/spec": ()=>{ syncInfRack(); renderSpec(); },
  "rl/mismatch": renderMismatch, "rl/lowprec": renderLowprec, "rl/qat": renderQat,
  "appendix/data": ()=>{ renderPublic(); renderReleases(); renderDB(); }, "appendix/quant": renderFormats, "appendix/models": renderModels, "appendix/method": ()=>{} };
const lastSub = {};
const viewKey = () => S.tab+(S.sub?"/"+S.sub:"");
function render(){
  try{ setWarn([]); VIEW[viewKey()](); }
  catch(e){ console.error(e); setWarn(["计算出错："+e.message]); }
}
function route(h){
  let p=String(h||"").replace(/^#/,"").split("/").filter(Boolean);
  if(p.length && !NAV[p[0]]){ const two=p.slice(0,2).join("/"), k=OLD[two]?two:OLD[p[0]]?p[0]:null;
    p = k ? OLD[k].split("/").concat(p.slice(k.split("/").length)) : []; }
  const page=NAV[p[0]]?p[0]:"overview";
  if(!NAV[page].length) return { page, sub:"", anchor:p[1]||"" };
  let sub=p[1]||"", anchor=p[2]||"";
  if(!NAV[page].includes(sub)){      // 不是子页名：当作锚点，找它所在的子页（例如旧的 #infer/tradeoff）
    const el=sub && ($(page+"-"+sub)||$(sub)), pn=el && el.closest("[data-pane]");
    if(pn && pn.dataset.pane.startsWith(page+"/")){ anchor=sub; sub=pn.dataset.pane.split("/")[1]; }
    else { anchor=""; sub=lastSub[page]||NAV[page][0]; } }
  return { page, sub, anchor };
}
function showTab(h){
  const {page,sub,anchor}=route(h), key=page+(sub?"/"+sub:""), same=viewKey()===key;
  S.tab=page; S.sub=sub; if(sub) lastSub[page]=sub;
  qa("nav.tabs a").forEach(a=>{ const t=a.dataset.tab; a.setAttribute("aria-current",t===page?"page":"false"); if(NAV[t].length) a.setAttribute("href","#"+t+"/"+(lastSub[t]||NAV[t][0])); });
  qa("nav.subtabs").forEach(n=>{ const on=n.dataset.page===page; n.hidden=!on; n.querySelectorAll("a").forEach(a=>a.setAttribute("aria-current",on&&a.dataset.sub===sub?"page":"false")); });
  qa("[data-pane]").forEach(p=>p.hidden=p.dataset.pane!==key);
  qa("[data-rack]").forEach(p=>p.hidden=!p.dataset.rack.split(" ").includes(key));
  try{ localStorage.setItem("moe-roof-nav3",JSON.stringify({h:key,last:lastSub})); }catch(e){}
  if(!(same && anchor)) render();
  const tgt=anchor && ($(sub+"-"+anchor)||$(page+"-"+anchor)||$(anchor));
  if(tgt && tgt.scrollIntoView) tgt.scrollIntoView({block:"start"});
  else if(!same){ const nav=document.querySelector("nav.tabs"); if(nav && nav.getBoundingClientRect().top<0) nav.scrollIntoView({block:"start"}); }
}
window.addEventListener("hashchange",()=>showTab(location.hash.slice(1)));
let first=location.hash.slice(1);
try{ const st=JSON.parse(localStorage.getItem("moe-roof-nav3")||"null"); if(st){ Object.assign(lastSub,st.last||{}); if(!first) first=st.h; } }catch(e){}
["q-ev-topic","q-ev-bw","q-ev-sort"].forEach(id=>$(id).addEventListener("change",render));
applyAll(); qShape(); buildEditor(); syncInputs(); showTab(first||"overview");
