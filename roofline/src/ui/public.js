/* ---------- 附录 · 数据与发布：数据点库里 origin = "public" 的点 ---------- */
const EVID = DB.set("quant_evidence");
const pubSrc = p => (p.url ? '<a href="'+esc(p.url)+'" target="_blank" rel="noopener">'+esc(p.src)+'</a>' : esc(p.src))+'<br><span class="tag">'+esc(p.date)+'</span>';
const isPareto = p => (p.tags||[]).includes("pareto");
function mokRatio(g){
  const t=DB.set("training_public").find(p=>p.replay&&p.replay.kind==="mok_ratio"); if(!t) return null;
  const R=t.replay, h=HW[R.hw], fl=2*R.n*R.K*3*R.De*R.F, tF=fl/(h.fp8*1e12*g.etaC);
  const mk=be=>Par.epLayer(be,{n:R.n,K:R.K,Z:R.Z,E:R.E,De:R.De,hw:h,topo:TOPO[R.topo],g,ac:false,prec:R.prec,imb:1.1,tExpF:tF*1.1,tExpB:2*tF*1.1,syncUs:60,mokF:0.25}).t;
  return { t, ratio:mk("deepep")/mk("mok") };
}
function renderPublic(){
  const g=S.glob, hw=HW[S.q.hw], inf=DB.set("inference_public");
  table($("pub-infer"),["模型 · 引擎 · 硬件","配置","数字","来源"],
    inf.filter(p=>!isPareto(p)).map(p=>[esc(p.model+" · "+p.engine+" · "+p.hw),esc(p.config),esc(p.value),pubSrc(p)]));
  table($("pub-pareto"),["对象","数字","为什么不能直接比","来源"],
    inf.filter(isPareto).map(p=>[esc(p.model==="多种"?p.config:p.model+" · "+p.config),esc(p.value),esc(p.caveat||""),pubSrc(p)]));
  const mk=mokRatio(g);
  table($("pub-train"),["对比 · 硬件","形状与设置","数字","本页","来源"],
    DB.set("training_public").map(p=>[esc(p.engine+" · "+p.hw),esc(p.model+"，"+p.config),esc(p.value),
      mk&&mk.t===p ? "一层 MoE F+B "+V(mk.ratio.toFixed(2)+"×")+'<br><span class="tag">同形状，<a href="#train/par/ep">后训练 · 并行 · EP</a> 的链路模型</span>' : "—",pubSrc(p)]));
  const topics=["全部",...new Set(EVID.map(r=>r.topic))], ts=$("q-ev-topic");
  if(ts.options.length!==topics.length){ ts.innerHTML=topics.map(t=>'<option>'+esc(t)+'</option>').join(""); ts.value="全部"; }
  const bwOnly=$("q-ev-bw").checked, srt=$("q-ev-sort").value, tp=ts.value;
  const rows=EVID.filter(r=>(tp==="全部"||r.topic===tp)&&(!bwOnly||r.arch==="bw")).sort((a,b)=>srt==="w"?b.w-a.w:(b.date.localeCompare(a.date)||b.w-a.w));
  const archN={bw:"Blackwell",bw120:"SM120",hopper:"Hopper",amd:"AMD",any:"—"};
  table($("q-evid"),["日期","来源","硬件","结论","数字","权重","本页模型"],
    rows.map(r=>[r.date+(r.dk==="read"?'<br><span class="tag">读取日期</span>':r.dk==="measured"?'<br><span class="tag">测量日期</span>':""),'<a href="'+esc(r.url)+'" target="_blank" rel="noopener">'+esc(r.src)+'</a><br><span class="tag">'+esc(r.topic)+'</span>',
      esc(r.hw)+'<br><span class="tag">'+archN[r.arch]+'</span>',esc(r.claim),esc(r.value),
      '<span class="wbar"><span style="width:'+(r.w*100).toFixed(0)+'%"></span></span>'+r.w.toFixed(2),r.chk?qCheck(r.chk,hw):"—"]));
}
