/* ================= 并行展开：FSDP / CP / EP。不另建模型，全部换参数重跑后训练核心的 Train.run ================= */
const CAT = { attn:"c-attn", proj:"c-lm", lin:"c-lin", moe:"c-moe", ep:"c-a2a", cp:"c-cp", fsdp:"c-fsdp", tp:"c-ar", sync:"c-sync", oth:"c-oth" };
const fK = n => n>=1048576 ? (n/1048576)+"M" : n>=1024 ? (n/1024)+"K" : String(n);
const pc = x => isFinite(x) ? (x*100).toFixed(Math.abs(x)<0.1?1:0)+"%" : "—";
const pow2upTo = n => { const a=[]; for(let v=1; v<=n; v*=2) a.push(v); return a; };

/* 浮动提示：内容一律走 textContent */
function showTip(ev, title, rows){
  const tip=$("tip"); tip.replaceChildren();
  if(title){ const t=document.createElement("div"); t.className="t0"; t.textContent=title; tip.append(t); }
  for(const r of rows){ const row=document.createElement("div"); row.className="row";
    if(r.c){ const i=document.createElement("i"); i.style.borderColor=css(r.c); row.append(i); }
    const b=document.createElement("b"); b.textContent=r.v; row.append(b);
    if(r.l){ const s=document.createElement("span"); s.textContent=r.l; row.append(s); }
    tip.append(row); }
  tip.hidden=false;
  const x=ev.pageX+14, w=tip.offsetWidth;
  tip.style.left=Math.max(4, Math.min(x, window.scrollX+document.documentElement.clientWidth-w-8))+"px"; tip.style.top=(ev.pageY+12)+"px";
}
const hideTip = () => { $("tip").hidden=true; };
function bindTips(svg){
  svg.onpointermove = ev => { const el=ev.target.closest("[data-tip]"); if(!el){ hideTip(); return; } const d=JSON.parse(el.getAttribute("data-tip")); showTip(ev,d.t,d.r); };
  svg.onpointerleave = hideTip;
}
const tipAttr = (t, r) => 'data-tip="'+esc(JSON.stringify({t, r}))+'"';

function axis(lo, hi, a, b, log){ if(log){ const L0=Math.log(lo), L1=Math.log(hi); return v => a + (Math.log(v)-L0)/(L1-L0)*(b-a); } return v => a + (v-lo)/(hi-lo)*(b-a); }
function niceTicks(lo, hi, n){ const span=hi-lo, s0=span/(n||5), p=Math.pow(10,Math.floor(Math.log10(s0))), s=[1,2,2.5,5,10].map(k=>k*p).find(k=>k>=s0)||p*10; const out=[]; for(let v=Math.ceil(lo/s)*s; v<=hi+1e-9*s; v+=s) out.push(+v.toPrecision(12)); return out; }
function logTicks(lo, hi, base){ const out=[], b=base||10; for(let v=Math.pow(b, Math.floor(Math.log(lo)/Math.log(b))); v<=hi*1.0001; v*=b) if(v>=lo*0.9999) out.push(v); return out; }

/* 折线图：series [{name, short, col, pts:[[x,y,{hollow,note}]]}]。默认十字准线吸附最近的 x；nearest 时吸附屏幕上最近的点（各线 x 不对齐的 Pareto） */
function lineChart(svgId, o){
  const svg=$(svgId), W=680, H=+svg.getAttribute("viewBox").split(" ")[3], m={l:64,r:18,t:16,b:44};
  const ok = p => p[1]!=null && isFinite(p[1]) && (!o.yLog || p[1]>0);
  const all = o.series.flatMap(s=>s.pts).filter(ok);
  if(!all.length){ svg.innerHTML='<text x="'+W/2+'" y="'+H/2+'" text-anchor="middle" class="lbl">'+esc(o.empty||"没有可画的点")+'</text>'; svg.onpointermove=null; return; }
  let x0=o.xMin ?? Math.min(...all.map(p=>p[0])), x1=o.xMax ?? Math.max(...all.map(p=>p[0]));
  let y0=o.yMin ?? (o.yLog ? Math.min(...all.map(p=>p[1]))/1.3 : 0), y1=o.yMax ?? Math.max(...all.map(p=>p[1]))*1.08;
  if(x0===x1){ x0*=0.5; x1*=2; } if(y0===y1) y1=y0*2||1;
  const X=axis(x0,x1,m.l,W-m.r,o.xLog), Y=axis(y0,y1,H-m.b,m.t,o.yLog), fx=o.xFmt||fN, fy=o.yFmt||fN;
  const xt = o.xLog ? logTicks(x0,x1,o.xBase||10) : niceTicks(x0,x1,6);
  let yt = o.yLog ? logTicks(y0,y1,10) : niceTicks(y0,y1,5); if(o.yLog && yt.length<3) yt=niceTicks(y0,y1,4).filter(v=>v>0);   // 不到一个数量级时改用线性刻度
  let s="";
  for(const v of yt) s+='<line class="gr" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+esc(fy(v))+'</text>';
  for(const v of xt) s+='<text x="'+X(v)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+esc(fx(v))+'</text>';
  s+='<line class="ax" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+(H-m.b)+'" y2="'+(H-m.b)+'"/>';
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+esc(o.xLab||"")+'</text><text x="12" y="'+(m.t-4)+'">'+esc(o.yLab||"")+'</text>';
  for(const v of (o.vlines||[])) if(v.x>=x0&&v.x<=x1){ const xv=X(v.x), rt=xv>W-130;     // 靠右的竖线把标注放到线的左下，免得出界或压住线尾的标注
    s+='<line x1="'+xv+'" x2="'+xv+'" y1="'+m.t+'" y2="'+(H-m.b)+'" style="stroke:var(--muted)" stroke-dasharray="4 4"/><text x="'+(rt?xv-4:xv+4)+'" y="'+(rt?H-m.b-6:m.t+10)+'"'+(rt?' text-anchor="end"':'')+'>'+esc(v.label)+'</text>'; }
  const direct = o.series.length<=5 && o.labelEnds!==false, ends=[];
  for(const se of o.series){ const pts=se.pts.filter(ok); if(!pts.length) continue;
    s+='<path d="'+pts.map((p,i)=>(i?"L":"M")+X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join("")+'" fill="none" style="stroke:'+css(se.col)+'" stroke-width="2" stroke-linejoin="round"/>';
    for(const p of pts){ const h=p[2]&&p[2].hollow; s+='<circle cx="'+X(p[0])+'" cy="'+Y(p[1])+'" r="'+(h?3.5:3)+'" style="fill:'+(h?"var(--panel)":css(se.col))+';stroke:'+css(se.col)+'" stroke-width="1.5"/>'; }
    if(direct){ const p=pts[pts.length-1]; ends.push({x:X(p[0]),y:Y(p[1])-7,t:se.short||se.name}); } }
  // 线尾标注：按线尾高低排好再往下错开 13，碰到下边再往上推，上下顺序和线一致
  const near=(a,b)=>Math.abs(a.x-b.x)<120;
  ends.sort((a,b)=>a.y-b.y).forEach((e,i)=>{ for(let j=0;j<i;j++) if(near(ends[j],e) && e.y<ends[j].y+13) e.y=ends[j].y+13; });
  for(let i=ends.length-1;i>=0;i--){ const e=ends[i]; e.y=Math.min(e.y,H-m.b-4); for(let j=i+1;j<ends.length;j++) if(near(ends[j],e) && e.y>ends[j].y-13) e.y=ends[j].y-13; }
  for(const e of ends){ const xr=e.x>W-130; s+='<text x="'+(xr?e.x-5:e.x+5)+'" y="'+e.y+'" text-anchor="'+(xr?"end":"start")+'" class="lbl">'+esc(e.t)+'</text>'; }
  s+='<line data-hair y1="'+m.t+'" y2="'+(H-m.b)+'" style="stroke:var(--ink);opacity:.35" hidden/><rect class="hit" x="'+m.l+'" y="'+m.t+'" width="'+(W-m.l-m.r)+'" height="'+(H-m.t-m.b)+'" style="fill:transparent"/>';
  svg.innerHTML=s;
  const xs=[...new Set(all.map(p=>p[0]))].sort((a,b)=>a-b), hair=svg.querySelector("[data-hair]");
  svg.onpointermove = ev => {
    const pt=svg.createSVGPoint(); pt.x=ev.clientX; pt.y=ev.clientY; const loc=pt.matrixTransform(svg.getScreenCTM().inverse());
    if(o.nearest){ let bp=null, bs=null, bd=Infinity;
      for(const se of o.series) for(const p of se.pts){ if(!ok(p)) continue; const dd=(X(p[0])-loc.x)**2+(Y(p[1])-loc.y)**2; if(dd<bd){ bd=dd; bp=p; bs=se; } }
      if(!bp){ hideTip(); return; }
      hair.setAttribute("x1",X(bp[0])); hair.setAttribute("x2",X(bp[0])); hair.removeAttribute("hidden");
      showTip(ev, bs.name, [{v:fy(bp[1]),l:o.yLab||"",c:bs.col},{v:fx(bp[0]),l:o.xLab||""}].concat(bp[2]&&bp[2].note?[{v:bp[2].note,l:""}]:[])); return; }
    let best=xs[0], bd=Infinity; for(const v of xs){ const dd=Math.abs(X(v)-loc.x); if(dd<bd){ bd=dd; best=v; } }
    hair.setAttribute("x1",X(best)); hair.setAttribute("x2",X(best)); hair.removeAttribute("hidden");
    const rows=o.series.map(se=>{ const p=se.pts.find(q=>q[0]===best); return p&&ok(p) ? {v:(o.tipFmt||fy)(p[1]), l:se.name+(p[2]&&p[2].note?"  "+p[2].note:""), c:se.col, y:p[1]} : null; }).filter(Boolean).sort((a,b)=>b.y-a.y);
    showTip(ev, (o.xLab||"x")+" = "+fx(best), rows.length?rows:[{v:"—",l:"这个位置没有点"}]);
  };
  svg.onpointerleave = () => { hair.setAttribute("hidden",""); hideTip(); };
}
function legendInto(id, items){ const el=$(id); if(!el) return;
  el.innerHTML=items.map(it=>it.line ? '<span><i style="border-color:'+css(it.col)+'"></i>'+esc(it.name)+'</span>' : '<span><span class="sw" style="background:'+css(it.col)+'"></span>'+esc(it.name)+'</span>').join(""); }

/* 甘特图：groups [{label, lanes, segs:[{lane,t0,t1,cat,label}]}] */
const LANE_NAME = { cpu:"CPU", sm:"SM（计算）", nvl:"NVLink", ib:"IB" };
function gantt(svgId, groups){
  const svg=$(svgId), W=680, laneH=20, gapG=16, m={l:92,r:12,t:10,b:28};
  let H=m.t+m.b; groups.forEach(gr=>{ H+=(gr.label?16:0)+gr.lanes.length*(laneH+3)+gapG; });
  svg.setAttribute("viewBox","0 0 "+W+" "+H);
  const tMax=Math.max(1e-9,...groups.flatMap(gr=>gr.segs.map(s=>s.t1))), X=axis(0,tMax,m.l,W-m.r);
  let s="", y=m.t;
  for(const v of niceTicks(0,tMax,6)) s+='<line class="gr" x1="'+X(v)+'" x2="'+X(v)+'" y1="'+m.t+'" y2="'+(H-m.b)+'"/><text x="'+X(v)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+esc(fT(v))+'</text>';
  for(const gr of groups){
    if(gr.label){ s+='<text x="4" y="'+(y+11)+'" class="lbl" style="font-weight:600">'+esc(gr.label)+'</text>'; y+=16; }
    const ly={};
    gr.lanes.forEach(k=>{ ly[k]=y; s+='<text x="'+(m.l-6)+'" y="'+(y+laneH/2+4)+'" text-anchor="end">'+esc(LANE_NAME[k]||k)+'</text><rect x="'+m.l+'" y="'+y+'" width="'+(W-m.l-m.r)+'" height="'+laneH+'" rx="3" style="fill:var(--soft)"/>'; y+=laneH+3; });
    for(const sg of gr.segs){ if(!(sg.lane in ly) || sg.t1-sg.t0<=0) continue;
      const x0=X(sg.t0), w=Math.max(1.5, X(sg.t1)-x0-2);
      s+='<rect x="'+(x0+1)+'" y="'+(ly[sg.lane]+1)+'" width="'+w+'" height="'+(laneH-2)+'" rx="3" style="fill:'+css(CAT[sg.cat]||"c-oth")+'" '+tipAttr(sg.label,[{v:fT(sg.t1-sg.t0),l:LANE_NAME[sg.lane]||sg.lane}])+'/>';
      if(w>70) s+='<text x="'+(x0+5)+'" y="'+(ly[sg.lane]+laneH/2+4)+'" style="fill:var(--panel);font-size:10.5px;pointer-events:none">'+esc(sg.label.length*6.4>w-8 ? sg.label.slice(0,Math.max(2,Math.floor((w-14)/6.4)))+"…" : sg.label)+'</text>'; }
    y+=gapG; }
  svg.innerHTML=s; bindTips(svg);
}
/* [lane,label,dur,cat,concurrent] 排进时间轴：concurrent 段和前一个非并发段同时开始 */
function layout(seq, t0){
  const segs=[]; let t=t0, lastStart=t0;
  for(const [lane,label,dur,cat,conc] of seq){ if(!(dur>0)) continue;
    if(conc){ segs.push({lane,label,t0:lastStart,t1:lastStart+dur,cat}); t=Math.max(t,lastStart+dur); continue; }
    segs.push({lane,label,t0:t,t1:t+dur,cat}); lastStart=t; t+=dur; }
  return { segs, end:t };
}
const GANTT_LEGEND = [["attention","attn"],["投影 / FFN","proj"],["线性注意力","lin"],["MoE GEMM","moe"],["EP 通信","ep"],["CP 通信","cp"],["TP 通信","tp"],["FSDP","fsdp"],["同步","sync"],["重算 / 其他","oth"]];
