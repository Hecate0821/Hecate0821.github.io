/* ================= 附录 · 模型结构 ================= */
function cacheParts(mm){ const dd=derive(mm), f=mm.full, l=mm.lin, sw=f.swa||0;
  return { kv: (f.n-sw)*(dd.kvTokElems*2 + (f.kind==="dsa"?(f.idxD+4)/(f.kpool||1):0)), st: l.n*(dd.linState*4 + dd.linConvDim*(l.conv-1)*2) + sw*dd.kvTokElems*2*(f.W||0) }; }
function renderModels(){
  const {m,d}=ctx(), keys=Object.keys(MODELS);
  table($("m-compare"),["模型","总参数（算）","激活（算）","官方","激活比例","注意力层","E / top-k","expert F（@latent）","KV / token（BF16，随上下文增长）","固定大小的状态 / 序列（线性 FP32、滑窗 KV BF16）"],
    keys.map(k=>{ const mm=k===S.modelKey?S.model:MODELS[k], dd=derive(mm), p=cacheParts(mm);
      return {cls:k===S.modelKey?"hl":"",cells:[esc(mm.short),fN(dd.total),fN(dd.active),esc(mm.official),(dd.active/dd.total*100).toFixed(1)+"%",
        mm.full.n+" "+mm.full.kind.toUpperCase()+(mm.full.swa?"（"+mm.full.swa+" 层滑窗 "+mm.full.W+"）":"")+(mm.lin.n?" + "+mm.lin.n+" "+mm.lin.kind:""),mm.moe.E?mm.moe.E+" / "+mm.moe.k+(mm.moe.Fs?" + shared":""):"dense",mm.moe.E?mm.moe.F+(mm.moe.Dlat?" @"+mm.moe.Dlat:""):"FFN "+mm.dense.F,fB(p.kv),p.st?fB(p.st):"—"]}; }),[1,2,4,6,7,8,9]);
  $("m-title").textContent=m.name; $("m-note").textContent=m.note+(m.mtp?" 另有 "+m.mtp+" 层 MTP（推理开投机时才加载）。":"");
  const cells=[], every = m.lin.n>0 ? Math.round(m.L/Math.max(1,m.full.n)) : 1; let left=m.full.n;
  const fs=m.full.swa||0, isSwa = i => fs>0 && i%(m.full.gEvery||4)!==(m.full.gOff||0);
  for(let i=0;i<m.L;i++){ const full = m.lin.n===0 || ((i+1)%every===0 && left>0) || (m.L-i<=left); if(full) left--; cells.push({full,swa:full&&isSwa(i),moe:i>=m.dense.n}); }
  $("m-stack").innerHTML=cells.map((c,i)=>'<span class="'+(c.moe?'moe':'')+'" style="background:'+(c.full?css("c-attn"):css("c-lin"))+(c.swa?';opacity:.45':'')+'" title="第 '+i+' 层：'+(c.swa?'滑窗 '+m.full.W+' ':'')+(c.full?m.full.kind.toUpperCase():m.lin.kind)+(c.moe?' + MoE':' + dense FFN')+'"></span>').join("");
  $("m-stack-legend").innerHTML='<span><span class="sw" style="background:var(--c-attn)"></span>'+m.full.kind.toUpperCase()+' 全注意力层 × '+(m.full.n-fs)+'</span>'+(fs?'<span><span class="sw" style="background:var(--c-attn);opacity:.45"></span>'+m.full.kind.toUpperCase()+' 滑窗 '+m.full.W+' 层 × '+fs+'</span>':'')+(m.lin.n?'<span><span class="sw" style="background:var(--c-lin)"></span>'+m.lin.kind+' 线性注意力层 × '+m.lin.n+'</span>':'')+'<span>白色小横条 = MoE 层（'+d.nMoe+'），没有横条 = dense FFN（'+m.dense.n+'）</span>';
  const pr=[["全注意力（每层，含复制部分）",d.pAttn,m.full.n],["　其中 MLA a 投影 / DSA indexer（推理各卡复制）",d.pAttnRep+d.pIdx,m.full.n],["线性注意力（每层）",d.pLin,m.lin.n],["一个 routed expert",d.pExpert,0],["routed experts（每层合计）",d.pRouted,d.nMoe],
    ["shared expert（每层）",d.pShared,d.nMoe],["latent 上下投影（每层）",d.pLat,d.nMoe],["router（每层）",d.pRouter,d.nMoe],["dense FFN（每层）",d.pDense,m.dense.n],["embedding",d.pEmb,1],["lm_head",d.pHead,1]];
  table($("m-params"),["部分","每份参数","份数","合计","占总参数"],
    pr.filter(r=>r[1]>0&&(r[2]>0||r[0].startsWith("一个"))).map(r=>{ const one=r[2]===0; return [r[0],fN(r[1]),one?"—":fInt(r[2]),one?"—":fN(r[1]*r[2]),one?"—":pct(r[1]*r[2],d.total)]; })
      .concat([{cls:"hl",cells:["<b>总计 / 每 token 激活</b>","","",fN(d.total),"激活 "+fN(d.active)]}]),[1,2,3,4]);
  // 缓存曲线
  const X0=128, X1=1048576, xs=logspace(X0,X1,140), series=[], colors=[css("c-attn"),css("c-lin"),css("c-moe"),css("c-a2a"),css("c-fsdp")], points=[];
  let ymax=0, ymin=Infinity;
  keys.forEach((k,i)=>{ const mm=k===S.modelKey?S.model:MODELS[k], p=cacheParts(mm);
    const pts=xs.map(n=>[n,(p.kv*n+p.st)/1e9]); pts.forEach(q=>{ ymax=Math.max(ymax,q[1]); ymin=Math.min(ymin,q[1]); });
    series.push({pts,color:colors[i%5],w:k===S.modelKey?3:1.4,dash:k===S.modelKey?null:"6 4"});
    if(p.st>0&&p.kv>0){ const ns=p.st/p.kv; if(ns>=X0&&ns<=X1) points.push({x:ns,y:2*p.st/1e9,color:colors[i%5],r:k===S.modelKey?5:3.5}); } });
  const cur=cacheParts(S.model), ci=keys.indexOf(S.modelKey);
  if(cur.st>0) series.push({pts:[[X0,cur.st/1e9],[X1,cur.st/1e9]],color:css("muted"),w:1.2,dash:"2 3",label:"线性注意力状态（常数）"});
  if(cur.kv>0&&cur.st>0){ series.push({pts:xs.map(n=>[n,cur.kv*n/1e9]),color:css("muted"),w:1.2,dash:"2 3"}); ymin=Math.min(ymin,cur.kv*X0/1e9); }
  plot($("m-cache"),{x0:X0,x1:X1,y0:Math.pow(10,Math.floor(Math.log10(Math.max(ymin,1e-4)))),y1:ymax*1.5,logx:true,logy:true,fx:fxPow,
    fy:v=>v>=1?v+" GB":v>=1e-3?+(v*1000).toPrecision(3)+" MB":+(v*1e6).toPrecision(3)+" KB",xl:"上下文长度（token）",yl:"每条序列的 KV + 状态",series,points,
    vlines:cur.st>0&&cur.kv>0?[{x:cur.st/cur.kv,color:colors[ci%5],label:"n* ≈ "+fN(cur.st/cur.kv)+" token：之后 KV 超过状态"}]:[]});
  $("m-cache-legend").innerHTML=keys.map((k,i)=>'<span style="color:'+colors[i%5]+'"><i'+(k===S.modelKey?'':' class="d"')+'></i>'+esc(MODELS[k].short)+'</span>').join("")+
    (cur.st>0?'<span style="color:var(--muted)"><i class="d"></i>'+esc(S.model.short)+' 的两个分量：状态（水平）与 KV（斜线）</span><span>圆点 = 各混合模型的交点 n*</span>':'<span>当前模型没有线性注意力层，缓存从一开始就和 n 成正比</span>');
}
