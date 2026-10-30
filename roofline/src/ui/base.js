/* ---------- 格式化 ---------- */
function fT(s){ if(!isFinite(s)) return "∞"; const a=Math.abs(s);
  if(a===0) return "0"; if(a<1e-3) return (s*1e6).toFixed(a<1e-5?1:0)+" µs"; if(a<1) return (s*1e3).toFixed(a<1e-2?2:1)+" ms";
  if(a<120) return s.toFixed(a<10?2:1)+" s"; return (s/60).toFixed(1)+" min"; }
function fB(b){ if(!isFinite(b)) return "∞"; const a=Math.abs(b); if(a>=1e12) return (b/1e12).toFixed(2)+" TB"; if(a>=1e9) return (b/1e9).toFixed(a>=1e11?0:a>=1e10?1:2)+" GB";
  if(a>=1e6) return (b/1e6).toFixed(1)+" MB"; if(a>=1e3) return (b/1e3).toFixed(1)+" KB"; return Math.round(b)+" B"; }
function fN(x){ if(!isFinite(x)) return "∞"; const a=Math.abs(x); if(a>=1e12) return (x/1e12).toFixed(2)+"T"; if(a>=1e9) return (x/1e9).toFixed(2)+"B";
  if(a>=1e6) return (x/1e6).toFixed(2)+"M"; if(a>=1e4) return (x/1e3).toFixed(1)+"K"; return (Math.round(x*10)/10).toLocaleString("en-US"); }
function fF(x){ return fN(x).replace(/([0-9.]+)([TBMK]?)$/, (m,a,b)=>a+" "+({T:"T",B:"G",M:"M",K:"K","":""}[b])+"FLOP"); }
function fI(x){ if(!isFinite(x)) return "∞"; return x>=100?Math.round(x).toString():x>=10?x.toFixed(1):x.toFixed(2); }
function fInt(x){ return isFinite(x)?Math.round(x).toLocaleString("en-US"):"∞"; }
const pct = (a,b) => b>0 ? (a/b*100).toFixed(a/b<0.1?1:0)+"%" : "—";
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;"); }
const css = n => "var(--"+n+")";

/* ---------- 画图 ---------- */
function niceStep(span,n){ const raw=span/n, p=Math.pow(10,Math.floor(Math.log10(raw))), e=raw/p; return (e>=7.5?10:e>=3.5?5:e>=1.5?2:1)*p; }
function plot(svg,c){
  try{ plotInner(svg,c); }catch(e){ svg.innerHTML='<text x="20" y="40" style="fill:var(--warn)">画图出错：'+esc(e.message)+'</text>'; console.error(e); }
}
function plotInner(svg,c){
  const W=680,H=c.h||340,m={l:68,r:18,t:16,b:46},iw=W-m.l-m.r,ih=H-m.t-m.b, L10=Math.log10;
  const sx=c.logx? v=>m.l+(L10(v)-L10(c.x0))/(L10(c.x1)-L10(c.x0))*iw : v=>m.l+(v-c.x0)/(c.x1-c.x0)*iw;
  const sy=c.logy? v=>m.t+ih-(L10(v)-L10(c.y0))/(L10(c.y1)-L10(c.y0))*ih : v=>m.t+ih-(v-c.y0)/(c.y1-c.y0)*ih;
  const id=svg.id+"C";
  let s='<defs><clipPath id="'+id+'"><rect x="'+m.l+'" y="'+m.t+'" width="'+iw+'" height="'+ih+'"/></clipPath>'+
    '<pattern id="'+id+'H" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="7" style="stroke:var(--warn)" stroke-width="1.4" opacity="0.35"/></pattern></defs>';
  const ticks=(a,b,log)=>{ const t=[]; if(log){ for(let e=Math.ceil(L10(a)-1e-9);e<=Math.floor(L10(b)+1e-9);e++) t.push(Math.pow(10,e)); if(t.length<3){ for(let e=Math.floor(L10(a));e<=Math.ceil(L10(b));e++) for(const k of [2,5]){const v=k*Math.pow(10,e); if(v>a&&v<b) t.push(v);} t.sort((x,y)=>x-y);} }
    else { const st=niceStep(b-a,6); for(let v=Math.ceil(a/st)*st; v<=b+1e-9*st; v+=st) t.push(+v.toPrecision(12)); } return t; };
  for(const v of ticks(c.x0,c.x1,c.logx)){ const x=sx(v); s+='<line class="gr" x1="'+x+'" x2="'+x+'" y1="'+m.t+'" y2="'+(m.t+ih)+'"/><text x="'+x+'" y="'+(m.t+ih+16)+'" text-anchor="middle">'+c.fx(v)+'</text>'; }
  for(const v of ticks(c.y0,c.y1,c.logy)){ const y=sy(v); s+='<line class="gr" x1="'+m.l+'" x2="'+(m.l+iw)+'" y1="'+y+'" y2="'+y+'"/><text x="'+(m.l-8)+'" y="'+(y+4)+'" text-anchor="end">'+c.fy(v)+'</text>'; }
  s+='<rect x="'+m.l+'" y="'+m.t+'" width="'+iw+'" height="'+ih+'" fill="none" class="ax"/>';
  s+='<text x="'+(m.l+iw/2)+'" y="'+(H-8)+'" text-anchor="middle" class="lbl">'+esc(c.xl)+'</text>';
  s+='<text transform="translate(14,'+(m.t+ih/2)+') rotate(-90)" text-anchor="middle" class="lbl">'+esc(c.yl)+'</text>';
  s+='<g clip-path="url(#'+id+')">';
  for(const r of (c.rects||[])){ const a=Math.max(m.l,sx(Math.max(r.x0,c.x0))), b=Math.min(m.l+iw,sx(Math.min(r.x1,c.x1))); if(b>a) s+='<rect x="'+a+'" y="'+m.t+'" width="'+(b-a)+'" height="'+ih+'" fill="url(#'+id+'H)"/>'; }
  for(const v of (c.vlines||[])){ if(!isFinite(v.x)||v.x<c.x0||v.x>c.x1) continue; const x=sx(v.x);
    s+='<line x1="'+x+'" x2="'+x+'" y1="'+m.t+'" y2="'+(m.t+ih)+'" style="stroke:'+v.color+'" stroke-width="1.2" stroke-dasharray="4 4"/>';
    if(v.label) s+='<text x="'+(x+4)+'" y="'+(m.t+12+(v.dy||0))+'" style="fill:'+v.color+'">'+esc(v.label)+'</text>'; }
  for(const se of c.series){ let d="",pen=false;
    for(const [x,y] of se.pts){ const ok=isFinite(x)&&isFinite(y)&&(!c.logy||y>0)&&(!c.logx||x>0);
      if(!ok){pen=false;continue} d+=(pen?"L":"M")+sx(x).toFixed(1)+" "+Math.max(-1e4,Math.min(1e4,sy(y))).toFixed(1); pen=true; }
    s+='<path d="'+d+'" style="fill:none;stroke:'+se.color+'" stroke-width="'+(se.w||2.2)+'"'+(se.dash?' stroke-dasharray="'+se.dash+'"':'')+' stroke-linejoin="round"/>';
    if(se.label && se.pts.length){ const last=se.pts.filter(p=>isFinite(p[0])&&isFinite(p[1])&&(!c.logy||p[1]>0)).slice(-1)[0];
      if(last){ const X=Math.min(sx(last[0]),m.l+iw-4), Y=Math.max(m.t+10,Math.min(m.t+ih-4,sy(last[1])-5)); s+='<text x="'+X+'" y="'+Y+'" text-anchor="end" style="fill:'+se.color+'">'+esc(se.label)+'</text>'; } } }
  s+='</g>';
  const placed=[];
  for(const p of (c.points||[])){ if(!isFinite(p.x)||!isFinite(p.y)||p.x<c.x0||p.x>c.x1||p.y<c.y0||p.y>c.y1) continue; const X=sx(p.x),Y=sy(p.y);
    const R=p.r||5, st=' style="fill:'+p.color+';stroke:var(--panel)" stroke-width="1.5"';
    if(p.shape==="sq") s+='<rect x="'+(X-R*0.85)+'" y="'+(Y-R*0.85)+'" width="'+(R*1.7)+'" height="'+(R*1.7)+'"'+st+'/>';
    else if(p.shape==="tri") s+='<polygon points="'+X+','+(Y-R*1.1)+' '+(X-R)+','+(Y+R*0.75)+' '+(X+R)+','+(Y+R*0.75)+'"'+st+'/>';
    else s+='<circle cx="'+X+'" cy="'+Y+'" r="'+R+'"'+st+'/>';
    if(p.label){ let ly=Y-8; while(placed.some(q=>Math.abs(q.y-ly)<12&&Math.abs(q.x-X)<130)) ly+=13; placed.push({x:X,y:ly});
      const right=X<m.l+iw-160; s+='<text x="'+(right?X+8:X-8)+'" y="'+ly+'" text-anchor="'+(right?"start":"end")+'" style="fill:'+p.color+'">'+esc(p.label)+'</text>'; } }
  svg.setAttribute("viewBox","0 0 "+W+" "+H); svg.innerHTML=s;
}
const fxPow = v => v>=1e6?(v/1e6)+"M":v>=1e3?(v/1e3)+"K":String(+v.toPrecision(3));
const fyFlops = v => v>=1e15?(v/1e15)+"P":v>=1e12?(v/1e12)+"T":v>=1e9?(v/1e9)+"G":String(v);
function logspace(a,b,n){ const r=[]; for(let i=0;i<n;i++) r.push(Math.pow(10,Math.log10(a)+i/(n-1)*(Math.log10(b)-Math.log10(a)))); return r; }

function stackBar(barEl, listEl, items, total, fmt){
  const sum=items.reduce((s,x)=>s+Math.max(0,x.v),0), T=Math.max(total||sum,1e-30);
  barEl.innerHTML=items.filter(x=>x.v>0).map(x=>'<span style="width:'+(x.v/T*100)+'%;background:'+x.color+'" title="'+esc(x.label)+'"></span>').join("");
  listEl.innerHTML=items.filter(x=>x.v>0).map(x=>'<div class="lab"><span class="sw" style="background:'+x.color+'"></span><span>'+esc(x.label)+(x.tag?' '+x.tag:'')+'</span></div><div class="num">'+fmt(x.v)+'</div><div class="num">'+pct(x.v,T)+'</div>').join("");
}
const pill = b => '<span class="pill '+b+'">'+b+'</span>';


const { MODELS, HW, TOPO, derive, dtB, peakOf } = Common;
const $ = id => document.getElementById(id);
const qa = s => Array.from(document.querySelectorAll(s));
const clone = o => JSON.parse(JSON.stringify(o));
const sum = (a,f) => a.reduce((s,x)=>s+(f?f(x):x),0);

const GROUPC = { attn:css("c-attn"), lin:css("c-lin"), moe:css("c-moe"), lm:css("c-lm"), misc:css("muted") };
const COMMC = { ar:css("c-ar"), latar:css("c-ar"), lmg:css("c-ar"), dpg:css("c-dp"), dps:css("c-dp"), disp:css("c-a2a"), comb:css("c-a2a"), cnt:css("c-a2a") };
const CLSC = { tp:css("c-ar"), cp:css("c-cp"), ep:css("c-a2a"), pp:css("c-dp"), dp:css("c-fsdp") };
const ENGC = { sglang:css("c-attn"), vllm:css("c-moe"), tokenspeed:css("c-lin"), titan:css("c-fsdp") };
const ENGN = { sglang:"SGLang", vllm:"vLLM", tokenspeed:"TokenSpeed", titan:"TitanRL" };
