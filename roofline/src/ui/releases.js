/* ---------- 附录 · 数据与发布：过去一年的发布与博客（数据点库的 releases 集）；RL 和投机两节的出处也从这里取链接 ---------- */
const REL = DB.set("releases");
const REL_N = { lmsys:"LMSYS", sglang:"SGLang", vllm:"vLLM", slime:"slime", miles:"Miles", areal:"AReaL" };
const REL_TAG = { rl:"RL", mismatch:"训推一致", quant:"低精度", spec:"投机解码", parallel:"并行", kernel:"kernel", model:"模型", hw:"硬件", api:"接口与调度", other:"其他" };
const REL_SUM = {
  lmsys:"SGLang 的 day-0 模型支持（DeepSeek-V4 / V4.1、Kimi K3、Qwen3.8、Inkling、Muse Spark Open 等）大多和 Miles 的 RL 支持一起发。推理：GB300 NVL72 / B300 上的 NVFP4、宽 EP 与 PD 分离、PP 长上下文 prefill、NVFP4 KV cache，以及 AMD、TPU、Trainium 后端；投机解码从 EAGLE-3（SpecBundle、SpecForge）走到 DFlash、Spec V2 和 DSpark。RL（Miles）：训推对齐（R3、TITO、Zero-KL）、低精度 RL（Unified FP8、INT4 QAT、MXFP8 / NVFP4、miniTE）、P2P 权重更新、OPD。",
  sglang:"v0.5.5 → v0.5.22，另有 Model Gateway 的 release。投机：Spec V2（overlap 调度）在 v0.5.11 成为默认、v0.5.13 弃用 V1，EAGLE-3 草稿模型（SpecBundle），DFlash（v0.5.11），DSpark（v0.5.16，v0.5.20 支持 PD 和 decode CP）。RL：确定性推理与 true on-policy（v0.5.5）、in-flight 权重更新（v0.5.6）、返回 routed experts 供 R3（v0.5.7）、采样掩码回放（v0.5.20）、session 制权重更新（v0.5.21）。低精度：NVFP4 MoE 与 KV cache、MXFP8 GEMM / MoE（v0.5.10）、online NVFP4（v0.5.14）。并行：DeepEP v2、DWDP、PP 长上下文。",
  vllm:"v0.11.1 → v0.32.0。Model Runner V2 逐步成为默认；async scheduling 在 v0.14.0 默认开启，v0.19.0 和投机解码零气泡重叠。投机：EAGLE-3、MTP、GPU n-gram、独立 draft、DFlash / DFlash2、DSpark、Dynamic SD。RL：batch-invariant kernel 逐步覆盖 torch.compile、FP8、NVFP4（训推逐位一致的基础），NCCL / IPC / P2P 权重同步 API（v0.16.0 起），routing replay，运行时更新 draft 权重。低精度：在线 MXFP8 / NVFP4 / MXFP4 量化，NVFP4 KV cache。",
  slime:"v0.2.0 → v0.4.0。训推一致：重要性采样自定义接口（MIS 等）和 R3 / R2 路由回放（v0.2.0），R3 扩到 DP-attention、DeepEP、MTP，FSDP 后端 true on-policy，top-p replay（v0.3.1），GLM-5 的 Megatron 与 SGLang 确定性对齐（v0.3.2），score centering（v0.4.0）。低精度：INT4 QAT（v0.2.2，v0.2.3 加非对称），block-FP8 rollout 权重可强制 UE8M0。权重同步：delta sync（v0.3.0）、并行 update_from_disk。另有 FSDP 后端、fully async 和 PipelineRL。",
  miles:"LMSYS 2025-11 首次介绍，v0.1.0 在 2026-08。训推对齐：R3、Zero-KL、TITO（多轮会话精确 tokenization）。低精度配方：NVFP4、MXFP4、MXFP8、FP8、INT4 QAT；v0.1.1 加 CuTe DSL 的 NVFP4 fake-QAT kernel（只量化 routed expert），v0.1.3 由 trainer 直接给出 MXFP8 rollout 权重。权重更新：P2P RDMA 和 disk-delta，v0.1.1 统一成一个 WeightUpdater，v0.1.3 重构 P2P。v0.1.2 加 score centering 和 torchtitan 后端；训练后的 MTP 权重同步到 draft。",
  areal:"v0.4.0 → v2.1.0。架构从 SPMD 换成 single controller（v1.0.0），v2.0.0 拆成训练 / 推理 / agent / 权重更新四类微服务；Archon 引擎（PyTorch 原生 5D 并行）和 FP8 blockwise 训练。训推一致：完整的 MIS / TIS（v1.0.1），统一的 rejection sampling（v1.0.4），IcePop / KPop 掩码（v2.0.0），接 SGLang 的 routed_experts 准备 routing replay（v1.0.0），记录每个 token 实际服务的权重版本（v2.1.0）。权重同步：FSDP 流水化、awex（NCCL P2P、共置 IPC）、AdamW delta。agent RL：OpenAI 兼容的 proxy、on-policy distillation。" };
const relOf = (p, v) => REL.find(r => r.project===p && r.version===v);
const aOut = (url, text) => '<a href="'+esc(url)+'" target="_blank" rel="noopener">'+esc(text)+'</a>';
function relLink(p, v, label){
  const r=relOf(p,v); if(!r) throw new Error("数据点库里没有这条发布："+p+" "+v);
  return aOut(r.url, label || (p==="lmsys" ? "LMSYS "+r.date : REL_N[p]+" "+v));
}
/* 公开数字按出处取：量化证据集按 url 正则，确定性一节按 id；库里没有就报错 */
function evOf(re){ const p=DB.set("quant_evidence").find(q=>re.test(q.url)); if(!p) throw new Error("数据点库里没有 "+re); return p; }
function detOf(id){ const p=DB.set("determinism_public").find(q=>q.id==="determinism_public:"+id); if(!p) throw new Error("数据点库里没有 "+id); return p; }
const evLink = (re, label) => { const p=evOf(re); return aOut(p.url, label||p.src); };
const detLink = (id, label) => { const p=detOf(id); return aOut(p.url, label||p.src); };
/* 精选条目 → 按日期排的表。条目是 [项目, 版本, 说明]，{ev: 正则, t: 说明}（量化证据集里 url 匹配的那条），或 {det: id, t: 说明} */
function ossTable(el, items){
  const rows=items.map(x=>{
    if(x.ev||x.det){ const p=x.ev?evOf(x.ev):detOf(x.det); return { date:p.date, src:aOut(p.url,p.src), t:x.t }; }
    const r=relOf(x[0],x[1]); if(!r) throw new Error("数据点库里没有这条发布："+x[0]+" "+x[1]);
    return { date:r.date, src:relLink(x[0],x[1],x[0]==="lmsys"?"LMSYS · "+r.title:null), t:x[2] }; });
  rows.sort((a,b)=>a.date.localeCompare(b.date));
  table(el,["日期","来源","内容"],rows.map(r=>['<span class="v">'+r.date+'</span>',r.src,esc(r.t)]));
}
function renderReleases(){
  const by={}; REL.forEach(r=>(by[r.project]=by[r.project]||[]).push(r));
  table($("rel-sum"),["项目",'<span class="nw">条数</span>',"时间","这一年的主线"],Object.keys(REL_N).map(p=>{ const ds=(by[p]||[]).map(r=>r.date).sort();
    return ['<span class="nw">'+REL_N[p]+(p==="lmsys"?" 博客":"")+'</span>',ds.length,ds.length?'<span class="nw">'+ds[0]+' →</span><br><span class="nw">'+ds[ds.length-1]+'</span>':"—",esc(REL_SUM[p])]; }),[1]);
  const ps=$("rel-proj"), ts=$("rel-tag");
  if(!ps.options.length){ ps.innerHTML='<option value="">全部</option>'+Object.keys(REL_N).map(p=>'<option value="'+p+'">'+REL_N[p]+'</option>').join("");
    ts.innerHTML='<option value="">全部</option>'+Object.keys(REL_TAG).map(t=>'<option value="'+t+'">'+REL_TAG[t]+'</option>').join(""); }
  const pv=ps.value, tv=ts.value, q=$("rel-q").value.trim().toLowerCase();
  const rows=REL.filter(r=>(!pv||r.project===pv)&&(!tv||r.tags.includes(tv))&&(!q||JSON.stringify(r).toLowerCase().includes(q)))
    .sort((a,b)=>b.date.localeCompare(a.date)||a.project.localeCompare(b.project));
  $("rel-n").textContent=rows.length+" / "+REL.length+" 条";
  table($("rel-list"),["日期","项目","发布","要点","标签"],rows.map(r=>['<span class="v">'+r.date+'</span>',REL_N[r.project],aOut(r.url,r.title||r.version),
    '<ul class="tight rel-hl">'+r.highlights.map(h=>'<li>'+esc(h)+'</li>').join("")+'</ul>',r.tags.map(t=>'<span class="tag">'+esc(REL_TAG[t]||t)+'</span>').join(" ")]));
}
["rel-proj","rel-tag"].forEach(id=>$(id).addEventListener("change",renderReleases));
$("rel-q").addEventListener("input",renderReleases);
