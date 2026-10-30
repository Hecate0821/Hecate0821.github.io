/* ---------- 和实测对照：数据点库里能复现的匿名实测，按同样的设置喂给 Train.run / Par ---------- */
/* 每个数据集一个 replay：pts 选出能复现的点，run 返回本页的预测（秒），meas 是实测（秒），rel = 只看组内相对快慢 */
const CALIB_BASE = { N:8, fw:"torchtitan", tp:1, pp:1, vpp:1, cp:8, ep:8, etp:1, linCP:"state", prec:"bf16", recompute:"none", cpLB:false, dispatcher:"standard", imb:1, etaA:273/989,
  offload:false, hsdp:1, gradDt:"fp32", reshard:true, hostSyncUs:30, mokF:0.25, workspaceGB:6, cpuAdamGBs:30, rcModules:[] };
const CALIB_BE = { local:"standard", nccl_a2a:"standard", deepep_legacy:"deepep", deepep_v2:"deepep" };
const BE_NAME = { local:"本地 expert", nccl_a2a:"NCCL a2a", deepep_legacy:"DeepEP legacy", deepep_v2:"DeepEP v2" };
const CALIB = [
  { set:"cp_attention", col:"s1", name:"单层注意力 F+B",
    note:"一层全注意力，Gtok = 8 / C 篇；本页 = 核心 F+B + 露出的 CP 通信",
    pts:()=>DB.where("cp_attention",{model:"qwen35",head_chunks:1}),
    label:p=>(p.method==="single"?"单卡":"CP"+p.cp+" "+Par.CP[p.method].name)+"，"+fK(p.seq),
    meas:p=>p.fb_ms/1e3,
    run:(p,run)=>{ const C=p.cp, r=run({S:p.seq,tok:p.seq/C,Gtok:p.seq*8/C,cp:C,cpType:C>1?p.method:"allgather"}), o=r.par.coreFullOp;
      return (o.fwd.t+o.dgrad.t)/o.layers+(r.par.cpFull?r.par.docs*r.par.cpFull.exposed:0); } },
  { set:"cp_full_model", col:"s2", name:"整模型 F+B（CP）",
    note:"EP = CP，NCCL a2a，full 重算；本页 = step − 优化器",
    pts:()=>DB.where("cp_full_model",{model:"qwen35"}),
    label:p=>"CP"+p.cp+" "+Par.CP[p.method].name+"，"+fK(p.seq),
    meas:p=>p.fb_s,
    run:(p,run)=>{ const r=run({S:p.seq,tok:p.seq/p.cp,Gtok:p.seq*8/p.cp,cp:p.cp,cpType:p.method,recompute:"full",ep:p.cp}); return r.stepT-r.tOpt; } },
  { set:"efsdp_full_model", col:"s3", name:"整模型一步（EP 后端）", rel:true,
    note:"CP8 AllGather，EP × expert FSDP = 8；DeepEP 两版都按本页的 expanded 算",
    pts:()=>DB.set("efsdp_full_model"),
    label:p=>BE_NAME[p.backend]+" EP"+p.ep+"，"+fK(p.seq),
    meas:p=>p.step_s,
    run:(p,run)=>{ const r=run({S:p.seq,tok:p.seq/8,Gtok:p.seq,cpType:"allgather",recompute:"full",ep:p.ep,dispatcher:CALIB_BE[p.backend]}); return r.stepT-r.tOpt; } },
  { set:"efsdp_comm", col:"s4", name:"DeepEP 四次交换",
    note:"不含 GEMM；本页另加 copy epilogue 的 HBM 读写",
    pts:()=>DB.set("efsdp_comm"),
    label:p=>BE_NAME[p.backend]+" EP"+p.ep+"，每卡 "+fK(p.seq/8),
    meas:p=>p.s,
    run:(p,run,g)=>{ const m=MODELS.qwen35, hw=HW.H200, n=p.seq/8, De=derive(m).De;
      const o=Par.epLayer("deepep",{n,K:8,Z:256,E:p.ep,De,hw,topo:TOPO.hgx8,g,ac:false,prec:"bf16",imb:1,syncUs:30});
      return o.tw+o.tAlpha+o.tSync+4*n*8*De*2/(hw.hbm*1e9*g.etaM); } },
  { set:"h200_kernels", col:"s5", name:"单卡 GEMM / HBM 拷贝",
    note:"纯 roofline：FLOP / (峰值 × 算力效率)，字节 / (HBM × 带宽效率)，不加 launch",
    pts:()=>DB.set("h200_kernels"),
    label:p=>p.op==="bf16_gemm"?"GEMM "+[p.m,p.n,p.k].map(fK).join("×"):"拷贝 "+fB(p.rw_bytes),
    meas:p=>p.ms/1e3,
    run:(p,run,g)=>{ const hw=HW.H200; return p.op==="bf16_gemm" ? 2*p.m*p.n*p.k/(hw.bf16*1e12*g.etaC) : p.rw_bytes/(hw.hbm*1e9*g.etaM); } },
];
let calibCache = null, calibSel = "cp_attention";
function calibReplay(g){
  const key=JSON.stringify(g); if(calibCache && calibCache.key===key) return calibCache.out;
  const run=o=>Train.run(MODELS.qwen35,HW.H200,TOPO.hgx8,Object.assign({},CALIB_BASE,o),g);
  const out=CALIB.map(c=>({ c, meta:DB.meta(c.set), rows:c.pts().map(p=>{ let pred=null; try{ pred=c.run(p,run,g); }catch(e){}
    const meas=c.meas(p); return { p, label:c.label(p), meas, pred, ratio:isFinite(pred)&&pred>0&&meas>0 ? pred/meas : null }; }).filter(x=>x.ratio!=null)
    .sort((a,b)=>a.label.split("，")[0].localeCompare(b.label.split("，")[0])||a.meas-b.meas) }));
  calibCache={key,out}; return out;
}
function quantile(a,q){ const s=[...a].sort((x,y)=>x-y), i=(s.length-1)*q, lo=Math.floor(i); return s.length ? s[lo]+(s[Math.ceil(i)]-s[lo])*(i-lo) : NaN; }

function renderCalib(){
  const {g}=ctx(), res=calibReplay(g);
  $("p-calib-sum").innerHTML='<thead><tr><th>数据集</th><th>点数</th><th>本页 / 实测（中位数）</th><th>中间 80%</th><th>±25% 以内</th><th>本页怎么算</th></tr></thead><tbody>'+
    res.map(({c,meta,rows})=>{ const rs=rows.map(x=>x.ratio), md=quantile(rs,0.5), inb=rs.filter(x=>x>=0.8&&x<=1.25).length;
      return '<tr data-calib="'+c.set+'"'+(c.set===calibSel?' class="hl"':'')+' style="cursor:pointer"><td><span class="dot" style="background:'+css(c.col)+'"></span>'+esc(c.name)+'<br><span class="tag">'+esc(c.set)+'</span></td><td class="num">'+rows.length+'</td><td class="num '+(c.rel?"":md>1.35||md<0.74?"bad":"ok")+'">'+md.toFixed(2)+'</td><td class="num">'+quantile(rs,0.1).toFixed(2)+' – '+quantile(rs,0.9).toFixed(2)+'</td><td class="num">'+(c.rel?"—":pc(inb/rs.length))+'</td><td>'+esc(c.note)+(c.rel?'；<b>只看组内顺序</b>':'')+'</td></tr>'; }).join("")+'</tbody>';
  calibScatter(res);
  legendInto("p-calib-legend",res.map(({c})=>({name:c.name,col:c.col})).concat([{name:"本页 = 实测（阴影 ±25%）",col:"muted",line:true}]));
  const sel=res.find(x=>x.c.set===calibSel)||res[0], meta=sel.meta;
  $("p-calib-cond").textContent="匿名数据点 · "+meta.gpus+" × "+meta.hw+" · "+meta.conditions;
  $("p-calib-pts").innerHTML='<thead><tr><th>设置</th><th>实测</th><th>本页</th><th>本页 / 实测</th></tr></thead><tbody>'+
    sel.rows.map(x=>'<tr><td>'+esc(x.label)+'</td><td class="num">'+fT(x.meas)+'</td><td class="num">'+fT(x.pred)+'</td><td class="num '+(sel.c.rel?"":x.ratio>1.35||x.ratio<0.74?"bad":"ok")+'">'+x.ratio.toFixed(2)+'</td></tr>').join("")+'</tbody>';
}
function calibScatter(res){
  const svg=$("p-calib-plot"), W=680, H=380, m={l:64,r:18,t:16,b:44};
  const all=res.flatMap(x=>x.rows), vs=all.flatMap(x=>[x.meas,x.pred]);
  const lo=Math.min(...vs)/1.5, hi=Math.max(...vs)*1.5, X=axis(lo,hi,m.l,W-m.r,true), Y=axis(lo,hi,H-m.b,m.t,true), tk=logTicks(lo,hi,10);
  let s="";
  for(const v of tk) s+='<line class="gr" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text x="'+(m.l-6)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+esc(fT(v))+'</text><text x="'+X(v)+'" y="'+(H-m.b+16)+'" text-anchor="middle">'+esc(fT(v))+'</text>';
  s+='<line class="ax" x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+(H-m.b)+'" y2="'+(H-m.b)+'"/>';
  s+='<polygon points="'+[X(lo)+","+Y(lo*1.25),X(hi/1.25)+","+Y(hi),X(hi)+","+Y(hi),X(hi)+","+Y(hi*0.8),X(lo/0.8)+","+Y(lo),X(lo)+","+Y(lo)].join(" ")+'" style="fill:var(--soft)"/>';
  s+='<line x1="'+X(lo)+'" y1="'+Y(lo)+'" x2="'+X(hi)+'" y2="'+Y(hi)+'" style="stroke:var(--muted)" stroke-dasharray="4 4"/>';
  s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-6)+'" text-anchor="middle">实测</text><text x="12" y="'+(m.t-4)+'">本页</text>';
  for(const {c,rows} of res) for(const x of rows){ const on=c.set===calibSel;
    s+='<circle cx="'+X(x.meas).toFixed(1)+'" cy="'+Y(x.pred).toFixed(1)+'" r="'+(on?4.5:3.5)+'" style="fill:'+css(c.col)+';stroke:var(--panel);opacity:'+(on?1:0.55)+'" stroke-width="1.5" '+
      tipAttr(c.name+"："+x.label,[{v:fT(x.meas),l:"实测"},{v:fT(x.pred),l:"本页",c:c.col},{v:x.ratio.toFixed(2),l:"本页 / 实测"}])+'/>'; }
  svg.setAttribute("viewBox","0 0 "+W+" "+H); svg.innerHTML=s; bindTips(svg);
}
$("p-calib-sum").addEventListener("click",e=>{ const tr=e.target.closest("[data-calib]"); if(!tr) return; calibSel=tr.dataset.calib; renderCalib(); });
