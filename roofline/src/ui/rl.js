/* ================= RL：训推不一致、低精度 rollout、QAT。rollout 照搬推理核心，训练照搬后训练核心 ================= */
const MM_PUB = [
  ["lmsys","2025-11-25-fp8-rl","BF16 训练 + FP8 rollout 时，MoE 从 30B 到 1T，TIS-clipfrac 和 log-prob 差一起涨；训练和 rollout 都用 FP8 后反而更小：前向的量化误差是主因"],
  ["lmsys","2026-04-25-deepseek-v4","DeepSeek-V4，32×GB300，FP8 rollout + BF16 训练，开 R3：首步 log-prob 差约 0.023"],
  ["lmsys","2026-06-04-nvidia-run-nemotron-3-ultra","Nemotron 3 Ultra：log-prob 平均绝对差约 0.01"],
  ["lmsys","2026-07-10-rocm-miles-dsv4","DeepSeek-V4-Flash 在 MI355X 上：修掉 hash 路由和 mHC 两处实现差异后约 0.09"],
  ["lmsys","2026-07-15-inkling-day0-support","975B Inkling 全参 GRPO + routing replay：train–rollout KL 约 10⁻³"],
  ["lmsys","2026-08-18-miles-v0-1","R3 + TITO；Zero-KL 下 log-prob 差可以是 0（目前只有 Qwen 3）"],
  ["lmsys","2026-09-10-deepseek-v41","DeepSeek-V4.1：训练前向复现推理端的 FP4 / FP8 舍入并用 routing replay，per-token KL 0.0012–0.0017"],
  {ev:/train-inference-consistency/,t:"训练前向直接调用 vLLM 的 batch-invariant 推理 kernel，反向 STE：每步 log-prob 差为 0，不再需要 TIS / R3；代价是 rollout 吞吐降到基线的 87.5–91.3%"},
  ["lmsys","2026-10-15-meta-muse-spark-open","Muse Spark Open：R3 把 log-prob 差从 0.015–0.020 降到 0.0044–0.0071"],
  ["lmsys","2026-10-16-beam","Beam：训练端对齐 SGLang 的数值（FP32 softcap / router），差减半"],
  ["lmsys","2026-10-25-minite","miniTE：严格对齐时 log-prob 差约 10⁻⁸"],
  ["lmsys","2026-10-22-miles-mi355x-agentic-rl","MI355X 上的 SWE agent RL：约 0.011"]];
const MM_OSS = [
  ["slime","v0.2.0","重要性采样自定义接口（MIS 等）；R3 与 R2 路由回放"],
  ["sglang","v0.5.6","RL 权重一致性检查"],
  ["slime","v0.2.1","R3 支持 DP-attention"],
  ["sglang","v0.5.7","返回每个 token 的 routed experts，供训练侧做 R3"],
  ["slime","v0.2.2","R3 完整支持 DeepEP 和 MTP；FSDP 后端的 MIS，修复 TIS / MIS"],
  ["areal","v1.0.0","接 SGLang 返回的 routed_experts，为 MoE routing replay 做准备"],
  ["areal","v1.0.1","完整的 MIS / TIS（此前只有 token 级 MIS）"],
  ["areal","v1.0.4","异步训练统一的 rejection sampling：ratio / KL 指标，token / 序列级，mask / clamp"],
  ["vllm","v0.21.0","routing replay 改用 device cache"],
  ["lmsys","2026-05-13-no-token-left-behind","Miles 的 TITO：第 n−1 轮的 prompt + response 必须是第 n 轮 prompt 的逐位前缀；重新分词、裁掉历史 reasoning、tool call 重渲染都会破坏它"],
  ["areal","v2.0.0","IcePop、KPop token 掩码并入 rejection sampling"],
  ["slime","v0.3.1","top-p masking，训练复现 rollout 的采样分布；R3 支持 PD 分离"],
  ["areal","v2.1.0","记录每个 token 实际服务的权重版本"],
  ["sglang","v0.5.20","return_sampling_mask：返回每步的采样支撑集和 log-prob，训练侧直接回放"],
  ["miles","v0.1.2","score centering loss：用 sampler 的 top-k 概率修正 off-policy 漂移，可选 TIS / MIS"],
  ["slime","v0.4.0","score centering，支持 top-p replay 和 IS 修正"]];
const DET_OSS = [
  ["sglang","v0.5.5","DeepSeek 架构的确定性推理（含 8×H200 上的 DeepSeek-V3）；true on-policy 模式 --rl-on-policy-target fsdp，对齐 FSDP 训练侧的数值"],
  ["vllm","v0.11.1","batch-invariant 支持 torch.compile，覆盖 Hopper / Blackwell 的 DeepGEMM 和 FlashInfer"],
  ["lmsys","2025-11-19-miles","Miles 的 true on-policy：FA3、DeepGEMM、batch-invariant kernel 加 torch.compile，训推 KL 为 0"],
  ["slime","v0.2.0","FSDP 后端 dense 模型 true on-policy；确定性 rollout 与训练可复现"],
  ["sglang","v0.5.6","Qwen3-Next 的确定性推理"],
  ["vllm","v0.12.0","batch-invariant BMM 优化：DeepSeek-V3.1 吞吐 +18.1%"],
  ["slime","v0.2.1","Qwen3-VL（dense）true on-policy"],
  ["vllm","v0.13.0","batch invariance 扩展到 FA2 和 LoRA"],
  ["vllm","v0.20.0","batch invariance 支持 NVFP4 linear"],
  ["vllm","v0.22.0","batch invariance 支持 CUTLASS FP8，端到端延迟改善 28.9%"],
  ["sglang","v0.5.18","修复 TP > 1 的 all-reduce 和 Gumbel 采样的确定性；GLM-4.7-Flash 支持确定性 FA4"],
  ["areal","v2.1.0","rollout 采样确定化"],
  ["vllm","v0.28.0","CUTLASS batch-invariant NVFP4 MoE"],
  ["slime","v0.3.2","GLM-5 的 Megatron 训练与 SGLang rollout 确定性对齐"],
  ["vllm","v0.29.0","确定性 decode replay；Blackwell 上 batch-invariant kernel 调优，端到端延迟降 33.6%"],
  ["sglang","v0.5.21","DeepEP v2 支持 batch-invariant 推理"],
  ["vllm","v0.31.0","batch invariance 默认用 breakable CUDA graph（不经 torch.compile），并禁用 SP 和 async TP"],
  ["vllm","v0.32.0","batch invariance 支持 custom all-reduce，扩展到 ROCm"]];
const LP_PUB = [
  ["lmsys","2025-11-25-fp8-rl","Unified FP8：训练和 rollout 都用 FP8，30B–235B MoE 上训推不一致低于“BF16 训练 + FP8 rollout”"],
  ["lmsys","2026-01-26-int4-qat","INT4 QAT + W4A16 rollout：稳定性和训推一致接近 BF16，约 1 TB 的模型单节点 H200 就能 rollout"],
  ["lmsys","2026-07-29-mxfp8-nvfp4-rl","Qwen3-30B-A3B、8×B200：五种低精度配置的 reward 曲线和 BF16 基本重合，rollout 更快；TE 和 FlashInfer 的量化器逐字节对齐，最后 15% 的层留在 BF16"],
  ["lmsys","2026-08-12-qwen3-8-day0-support","64 张 GB300 上 BF16 Megatron 与 NVFP4 rollout 共置做 LoRA RL，train / rollout KL 平稳"],
  ["lmsys","2026-10-25-minite","expert 按 token NVFP4 前向 + BF16 反向，8×B300 learner 更新约 1.33–1.37× BF16；但 BF16 → NVFP4 的转换 + 同步把每步多出约 40 s（见下面的权重同步）"]];
const WS_PUB = [
  ["sglang","v0.5.6","in-flight 权重更新；MTP 层支持 update_weights_from_tensor"],
  ["vllm","v0.16.0","原生 NCCL 权重同步 API；QeRL 逐层权重重载"],
  ["vllm","v0.17.0","基于 IPC 的权重同步 API"],
  ["areal","v1.0.3","FSDP 权重同步流水化：上一个 bucket 的广播和下一个 bucket 的 all-gather 重叠"],
  ["lmsys","2026-04-29-p2p-update","Mooncake TransferEngine 的 RDMA P2P：1T 的 Kimi-K2 权重更新 53 s → 7.2 s；EP 小时 CPU 副本的加载开销会超过收益"],
  ["areal","v1.0.4","awex 后端（NCCL P2P），支持 FSDP → SGLang 和 Megatron PP / CP / EP"],
  ["vllm","v0.21.0","/start_weight_update 与 /finish_weight_update"],
  ["slime","v0.3.0","delta weight sync（disk 与 NCCL 两种传输）"],
  ["slime","v0.3.1","并行 update_from_disk、disk 级 delta、engine 侧 /pull_weights"],
  ["miles","v0.1.0","P2P RDMA：1T 模型更新 53.3 s → 7.2 s；disk-delta 的载荷 62.4 GB → 0.69–0.83 GB"],
  ["areal","v2.1.0","分离部署下的 AdamW delta 传输：首次全量，之后只传稀疏增量"],
  ["vllm","v0.29.0","sharded_rdt P2P（worker 只拉自己的 TP / EP 分片）、rank-local IPC、稀疏更新"],
  ["miles","v0.1.1","权重更新统一成一个 WeightUpdater：broadcast / P2P / RDT / disk-delta / 共置"],
  ["sglang","v0.5.21","权重更新改成 session 制（begin / end_weight_update），可以连 draft 一起更新"],
  ["miles","v0.1.3","P2P 重构：Qwen3-30B-A3B 上每个 sender 注册的显存 28.45 → 2.00 GiB，首次更新 99.5 → 3.9 s"],
  {ev:/minite/,t:"在线 RL 每步 BF16 → NVFP4 的转换 + 同步 9.94 → 55.26 s：rollout 的收益被抵掉"}];
const QAT_OSS = [
  ["slime","v0.2.2","INT4 QAT（含 int4_quant CUDA kernel）"],
  ["lmsys","2026-01-26-int4-qat","INT4 QAT RL 全流程：训练 fake quant + STE，rollout W4A16；方案同步到 slime 和 Miles"],
  ["slime","v0.2.3","INT4 QAT 支持非对称量化"],
  {ev:/humansand/,t:"The 4-bitter Lesson：NVFP4 前向 + BF16 反向时，dequantized 的梯度对应实际执行的函数，Adam 下大尖峰少得多；4/6 用到 RL 的权重和激活上"},
  ["lmsys","2026-07-29-mxfp8-nvfp4-rl","Miles 的 Blackwell 原生配方：端到端 MXFP8，以及 MoE expert 按 token 的 NVFP4"],
  ["miles","v0.1.0","低精度训练配方：NVFP4、MXFP4、MXFP8、FP8、INT4 QAT"],
  ["slime","v0.3.2","ROCm 上的 INT4 QAT kernel"],
  ["miles","v0.1.1","融合的 CuTe DSL NVFP4 fake-QAT kernel，只量化 routed expert"],
  {ev:/train-inference-consistency/,t:"训练前向直接调用推理 kernel（block FP8 W8A8、NVFP4 W4A16），反向 STE：同样是“前向跑 rollout 的量化函数”，只是连 kernel 一起复用"},
  ["lmsys","2026-10-25-minite","miniTE：反向用高精度参考梯度检验，选原始值（high_precision）"]];

function renderMismatch(){
  const {m,d}=ctx(), Sq=S.trn.S, e=m.moe, f=m.full, nL=e.E?d.nMoe:m.L, idB=e.E>256?2:1;
  const kvTok=Math.max(0,f.n-(f.swa||0))*d.kvTokElems*2, r3=e.E?d.nMoe*e.k*idB:0;
  const R=[["token id（TITO）",4,"int32；多轮 agent 里直接把 rollout 的 token 交给训练，不重新分词"],
    ["采样时的 log-prob",4,"FP32，算重要性比值 π_train / π_rollout"],
    ...(e.E?[["R3 路由（"+d.nMoe+" 层 × top-"+e.k+"）",r3,"每层 k 个 expert id，按 "+(idB*8)+" 位存；实现里常用 int32，则 × "+(4/idB)]]:[]),
    ["采样掩码（top-p 支撑集，按 20 个候选）",160,"每个候选一个 id + 一个 log-prob，训练侧回放同样的截断"]];
  const tot=sum(R,x=>x[1]);
  $("r-mm-lede").innerHTML="按当前模型（"+esc(m.short)+"）算，每条序列取后训练页的平均序列长度 S = "+fInt(Sq)+"。修正和对齐需要 rollout 交回来的东西都很小："+(e.E?"R3 是 KV cache 的 "+V(pct(r3,kvTok))+"，":"")+"贵的不在搬运，在训练侧要按给定的路由和采样走，以及两边 kernel 的对齐。";
  table($("r-mm"),["内容","每 token","每条序列","说明"],R.map(x=>[esc(x[0]),fB(x[1]),fB(x[1]*Sq),esc(x[2])]).concat([
    {cls:"hl",cells:["<b>合计</b>",fB(tot),fB(tot*Sq),""]},
    ["对照：KV cache（BF16）",fB(kvTok),fB(kvTok*Sq),"推理侧本来就存着"],
    ["对照：每个"+(e.E?" MoE ":"")+"层输入的隐状态（BF16）",fB(nL*m.D*2),fB(nL*m.D*2*Sq),"如果改成回放激活要搬的量"]]),[1,2]);
  ossTable($("r-mm-pub"),MM_PUB);
  ossTable($("r-mm-oss"),MM_OSS);
}

/* 确定性 rollout：哪些地方随 batch 变、四个层级、代价、和低精度的关系。数字都从 determinism_public / quant_evidence / releases 取 */
function renderDeterminism(){
  const {m,hw,topo,g}=ctx(), f=Math.min(1,Math.max(0,S.q.roll)), D=m.D, rAR=Math.floor(512*1024/(2*D)), e=m.moe;
  const u=detOf("tml-unique"), tc=detOf("tml-cost"), lc=detOf("lmsys-det-cost"), ev=evOf(/train-inference-consistency/), vt=detOf("vllm-tt-bitwise");
  $("r-det-lede").innerHTML="GPU 上同一个 kernel、同样的输入跑两遍，结果本来就一样：前向里几乎没有原子加。推理服务不确定，是因为一个请求和哪些请求拼进同一个 batch 随负载变；batch 一变，kernel 就换分块、split-K、split-KV 或通信算法，"+
    "同一个请求的累加顺序跟着变，浮点加法不满足结合律，结果差在最后几位，几十个 token 以后采样就分叉。"+detLink("tml-unique","Thinking Machines Lab")+" 把它叫作缺少 batch invariance：Qwen3-235B 在 temperature 0 下同一个 prompt 跑 1000 次，得到 "+V(u.nDefault)+" 种输出，换成 batch-invariant kernel 后只有 "+u.nInvariant+" 种。"+
    "下表是一层里会随 batch 变的地方，通信一行按当前模型（"+esc(m.short)+"，D = "+fInt(D)+"）算。";
  table($("r-det-ops"),["随 batch 变的地方","为什么变","batch-invariant 的做法","出处"],[
    ["RMSNorm 等逐行归约","batch 小时一行拆给几个 core 并行归约，batch 大时一行一个 core","不管 batch 多大，一行都由一个 core 归约完；小 batch 时并行度不够也不拆",detLink("tml-unique","TML")],
    ["GEMM","小 M 时用 split-K / stream-K 填满 SM，不同 M 选不同的 tile 和指令","所有 M 用同一套 tile、不 split-K；TML 的实现比 cuBLAS 慢约 20%",detLink("tml-unique","TML")+"；"+relLink("vllm","v0.11.1")+"、"+relLink("vllm","v0.12.0")],
    ["attention","decode 的 split-KV 按“固定份数”拆，每份长度随上下文和 batch 变；prefill 的 chunk 切法、prefix cache 命中与否改变 KV 的分块","按“固定每份长度”拆；先把新 token 写进 KV cache 再做 attention；chunked prefill 的边界对齐 split 长度",detLink("tml-unique","TML")+"；"+detLink("lmsys-det-cost","SGLang")],
    ["TP all-reduce","按消息大小换算法：本页的推理核心在每步 token ≤ "+V(fInt(rAR))+"（消息 ≤ 512 KB）时走 one-shot，超过走 two-shot，两种算法的累加顺序不同；融合 AR + RMSNorm 的 kernel 也只在 token 数不超过阈值时启用。SP / async TP 把 AR 拆成 RS + AG、按 token 数切块和 GEMM 重叠","固定算法和累加顺序；不同 TP 大小用同一棵归约树（TBIK）；vLLM 在 batch invariance 下禁用 SP 和 async TP",
      relLink("sglang","v0.5.18")+"、"+relLink("vllm","v0.31.0")+"、"+relLink("vllm","v0.32.0")+"；"+detLink("tbik","TBIK")],
    ["MoE",e.E?"每个 expert 拿到的行数随 batch 变（当前模型 E = "+e.E+"，top-"+e.k+"），grouped GEMM 跟着换 tile；EP 的 combine 把 top-"+e.k+" 个 expert 的输出加起来，顺序随到达顺序变":"当前模型是 dense，没有这一项；MoE 模型里每个 expert 的行数随 batch 变，EP combine 的累加顺序随到达顺序变",
      "固定 tile；combine 按固定顺序累加。SGLang 首个版本只支持 dense 模型",relLink("sglang","v0.5.21")+"、"+relLink("vllm","v0.28.0")+"；"+detLink("lmsys-det-limit","LMSYS")],
    ["激活量化","按张量的动态 scale 取整个 batch 的 amax，别的请求里的 outlier 会改变这个请求的量化结果","按 token、按 1D 块的 scale 只看自己这一行，天然 batch-invariant；或者用校准好的静态 scale",'<a href="#appendix/quant">量化基础</a>'],
    ["投机解码的验证","验证是宽度 k + 1 的 forward，GEMM 行数和 attention 的 query 数都和普通 decode 不同","同一个 token 在宽度 1 和宽度 k + 1 的 forward 里结果要一样",'<a href="#infer/spec">投机解码</a>'],
    ["采样","随机数按 batch 里的位置取，换了位置就换了随机数","每个请求一个种子（SGLang 用 Gumbel 噪声实现带种子的 multinomial）",detLink("lmsys-det-cost","SGLang")+"；"+relLink("sglang","v0.5.18")]]);
  table($("r-det-lv"),["层级","保证什么","在上一层之外还要做什么","公开结果"],[
    ["1 确定性 rollout","同一个请求、同一个种子，不管负载怎么变输出都一样；rollout 可以重放，出了问题能复现","上表各行的 batch-invariant kernel 和按请求的种子",detLink("tml-unique","TML")+"："+u.nDefault+" → "+u.nInvariant+" 种输出；"+relLink("sglang","v0.5.5")+"、"+relLink("vllm","v0.11.1")],
    ["2 可复现 RL","同样的配置重跑，整条训练曲线重合","训练侧也确定：Megatron --deterministic-mode、反向用 FA2、NCCL / cuBLAS / TE 的确定性环境变量",detLink("lmsys-slime-repro","slime + SGLang")+"：Qwen3-8B 两次曲线完全重合；"+relLink("slime","v0.3.2")+" 扩到 GLM-5"],
    ["3 训推逐位一致（true on-policy）","训练算出的 log-prob 和 rollout 逐位相等，KL = 0，不再需要 TIS / R3","训练前向直接调用推理的 kernel（或推理调用训练的），反向自己写或用 STE",
      detLink("vllm-tt-bitwise","vLLM + TorchTitan")+"：KL 始终 0，整体慢 "+vt.slow+"×；"+aOut(ev.url,"vLLM 2026-10")+"：DeepSeek-V4-Flash 上 log-prob 差 0；"+relLink("lmsys","2025-11-19-miles","Miles")+"；"+relLink("sglang","v0.5.5")+" 的 --rl-on-policy-target fsdp"],
    ["4 跨 TP 大小也一致","推理和训练的 TP 大小不同（比如推理 TP 8、训练 FSDP）时也逐位相等","本地 GEMM 和跨卡 collective 用同一棵固定的二叉归约树",detLink("tbik","TBIK")+"：vLLM 和 FSDP 之间逐位相同"]]);
  // 代价
  const sp=x=>Array.isArray(x)?x.map(v=>f*v+(1-f)):[f*x+(1-f)], fx=a=>{ const lo=Math.min(...a), hi=Math.max(...a); return "×"+(lo===hi?lo.toFixed(2):lo.toFixed(2)+"–"+hi.toFixed(2)); };
  const dec=bi=>Infer.decode(m,hw,topo,Object.assign({},S.inf,{phase:"decode",bi,specK:0}),g), sBi=dec(true).stepT/dec(false).stepT;
  $("r-det-cost-lede").innerHTML="只有 rollout 变慢时，RL 一步的时间按 Amdahl 是 f·s + (1 − f)，f = "+f+" 是 BF16 时 rollout 占一步的比例（和低精度 rollout 一节共用）。开销这一年降得很快："+
    relLink("vllm","v0.12.0")+" 的 batch-invariant BMM 让吞吐 +18.1%，"+relLink("vllm","v0.22.0")+" 接上 CUTLASS FP8 后端到端延迟改善 28.9%，"+relLink("vllm","v0.29.0")+" 在 Blackwell 上调优后再降 33.6%。CUDA graph 也必须能开："+detLink("lmsys-det-graph","SGLang 确定性模式")+"下 Qwen3-8B 开 graph 从 441.73 涨到 1245.51 tok/s。";
  const C=[
    ["SGLang 确定性模式",detLink("lmsys-det-cost","LMSYS 2025-09"),"平均慢 "+((lc.slow-1)*100).toFixed(2)+"%，多数 25–45%","rollout",sp(lc.slow)],
    ["TML 的 batch-invariant kernel：未优化 / 改进 attention 后",detLink("tml-cost","TML 2025-09"),"Qwen3-8B："+tc.tBase+" s → "+tc.tDet+" s / "+tc.tDetOpt+" s","rollout",sp([tc.tDet/tc.tBase,tc.tDetOpt/tc.tBase])],
    ["训练前向调用推理 kernel（第 3 层）",aOut(ev.url,"vLLM 2026-10"),"rollout 吞吐是基线的 "+ev.thr.map(x=>(x*100).toFixed(1)+"%").join(" / "),"rollout；训练侧没单独报",sp(ev.thr.map(x=>1/x))],
    ["TorchTitan 复用 vLLM 的算子（第 3 层）",detLink("vllm-tt-bitwise","vLLM 2025-11"),"整体慢 "+vt.slow+"×","整个 RL",[vt.slow]],
    ["本页推理核心的 batch-invariant 开关：所有 kernel × "+S.inf.biFactor,'<a href="#infer/step">推理 · 一步</a>',"当前 decode 配置一步 ×"+sBi.toFixed(2)+(sBi<S.inf.biFactor-0.01?"：跨卡通信不变":"：当前配置没有跨卡通信，整步一起变慢"),"rollout",sp(sBi)]];
  table($("r-det-cost"),["做法","出处","报告的开销","算在哪一侧","RL 一步的时间（f = "+f+"）"],C.map(x=>[esc(x[0]),x[1],esc(x[2]),esc(x[3]),fx(x[4])]),[4]);
  // 和低精度的关系
  const qh=HW[S.q.hw], r16=qInferE2E(qh,"bf16"), r4=qInferE2E(qh,"nvfp4"), sr=r16.stepT/r4.stepT;
  $("r-det-lp").innerHTML=ul([
    "<b>量化逐位一致是必要条件，不是充分条件。</b><a href=\"#rl/lowprec\">低精度 rollout</a> 要求训练前向和 rollout 量化出同样的 FP4 码和 scale；但两边的 GEMM 累加顺序、attention 和通信仍然不同，log-prob 还有差。要 KL = 0 得再做到第 3 层："+aOut(ev.url,"vLLM 2026-10")+" 让训练前向直接调用 block FP8 W8A8、NVFP4 W4A16 的推理 kernel。",
    "<b>scale 的粒度决定量化本身是不是 batch-invariant。</b>按 token 的 scale 和 1D 块的 scale（block FP8 激活 1×128、MX 1×32、NVFP4 1×16）只看这一行；按张量的动态 scale 取整个 batch 的 amax，别的请求会改变这个请求的量化结果。NVFP4 第二级 FP32 scale 按张量取时也是这样，所以 RL rollout 用按 token 的那种。激活不用 2D 方块，原因之一也是它让 128 个 token 共享一个 scale（见<a href=\"#train/quant\">低精度训练</a>的 1D / 2D 表）。",
    "<b>低精度 kernel 要各自有 batch-invariant 版本。</b>"+relLink("vllm","v0.20.0")+" 的 NVFP4 linear、"+relLink("vllm","v0.22.0")+" 的 CUTLASS FP8、"+relLink("vllm","v0.28.0")+" 的 NVFP4 MoE、"+relLink("sglang","v0.5.21")+" 的 DeepEP v2：每换一种格式或后端，这件事都要重做一遍。",
    "<b>两笔账放在一起算。</b>按本页的模型，"+esc(qh.name)+" 上 NVFP4 rollout 的 decode 一步比 BF16 快 "+V(fX(sr))+"（<a href=\"#infer/quant\">低精度推理</a>）；再开 SGLang 确定性模式（平均 ×"+lc.slow.toFixed(2)+"），净 "+V(fX(sr/lc.slow))+"。低精度换来的速度和确定性的开销是同一量级。"]);
  table($("r-det-pub"),["日期","来源","主题","模型 · 硬件","设置","数字"],DB.set("determinism_public").slice().sort((x,y)=>x.date.localeCompare(y.date))
    .map(p=>['<span class="v">'+esc(p.date)+'</span>','<span class="nw">'+aOut(p.url,p.src)+'</span>'+(p.code?"<br>"+aOut(p.code,"代码"):""),'<span class="nw">'+esc(p.what)+'</span>',esc(p.model)+(p.hw&&p.hw!=="—"?"<br>"+'<span class="tag">'+esc(p.hw)+'</span>':""),esc(p.setting),esc(p.value)]));
  ossTable($("r-det-oss"),DET_OSS);
}

function renderLowprec(){
  const {hw,f}=qRL(), {m,d,topo}=ctx(), hwS=HW[S.hw];
  $("r-lp-lede").innerHTML="rollout 照搬推理页的 decode 配置（"+esc(ENGN[S.inf.engine]||S.inf.engine)+"，N = "+S.inf.N+"，B = "+S.inf.B+"），训练照搬后训练页的配置，GPU 都换成 "+esc(hw.name)+"，只换精度。RL 一步按 Amdahl 合成："+
    V("1 / (f / s_rollout + (1 − f) / s_train)")+"，f = "+f+" 是 BF16 时 rollout 占一步的比例（左侧可改）。最后一列是能不能做到两侧量化结果逐位一致，见下面的生产链路。";
  ossTable($("r-lp-pub"),LP_PUB);
  // 权重同步：一次全量要搬的字节
  const lin=d.total-d.pEmb-d.pHead, ex=d.pExpertsAll, rest=d.total-ex, emb=(d.pEmb+d.pHead)*2;
  const F=[["BF16",d.total*2,"默认"],["FP8 block（线性层）",lin*Common.dtB("fp8")+emb,"FP8 rollout（Unified FP8；DeepSeek-V4 day-0）"],["MXFP8（线性层）",lin*Common.dtB("mxfp8")+emb,"Miles 的 MXFP8 配方（v0.1.3 起 trainer 直接给出 rollout 权重）"],
    ...(ex?[["NVFP4（只量化 routed expert）",ex*Common.dtB("nvfp4")+rest*2,"Miles 的 NVFP4 rollout"],["INT4 W4A16（只量化 routed expert，group 128）",ex*(0.5+2/128)+rest*2,"slime / Miles 的 INT4 QAT"]]:[])];
  const ib=hwS.ib*1e9, N=topo.gpus;
  $("r-ws-lede").innerHTML="按当前模型（"+fN(d.total)+" 参数）算一次全量同步要搬的字节，链路按 "+hwS.name+" 每卡 IB "+hwS.ib+" GB/s（rollout 在另一组机器上；共置时走 NVLink / CUDA IPC，快近一个数量级）。"+
    "低精度 rollout 的权重还要先在训练侧量化一遍；字节省下来，转换本身可能成为新的瓶颈。连续两步之间权重变化很小，delta 同步只传变了的部分。";
  table($("r-ws"),["rollout 权重格式","每次同步","相对 BF16","一张卡按 IB 发完","训练侧 "+N+" 张卡各发一份分片","谁这么做"],F.map(([n,b,w])=>[esc(n),fB(b),pct(b,F[0][1]),fT(b/ib),fT(b/(N*ib)),esc(w)]),[1,2,3,4]);
  ossTable($("r-ws-pub"),WS_PUB);
}

function renderQat(){
  const {q,hw,g,Q}=qHead(), N=q.N, K=q.K, Mt=Math.max(16,q.Mt), to={nMicro:q.nMicro,tokFused:q.tokFused};
  const G=(k,o)=>Q.trainGemm(hw,g,k,Mt,N,K,Object.assign({},to,o)), b=G("bf16");
  const fq=Math.max(g.kernelUs*1e-6, N*K*(2+2)/(hw.hbm*1e9*g.etaM))/Math.max(1,q.nMicro);   // fake quant：读 BF16 权重、写回量化再反量化的 BF16，按 micro-batch 摊
  $("r-qat-lede").innerHTML="一个线性层一个 micro-batch 的三遍 GEMM（fprop + dgrad + wgrad）加量化相关 kernel，形状 N = "+fInt(N)+"、K = "+fInt(K)+"、M = "+fInt(Mt)+"，"+esc(hw.name)+"。显式 fake quant 按每个 micro-batch 读写一遍权重估，反向 STE 直通不多算。前两行以外都是真低精度 GEMM。";
  const rows=[
    ["BF16 训练 + 训练后量化 rollout","BF16","BF16","没量化的 y = x·w","否",b,"多数默认配置"],
    ["显式 fake quant + STE","BF16（权重先量化再反量化）","BF16，STE 直通","量化后的函数","权重逐位一致；激活看 rollout 是否量化",{t:b.t+fq,extra:b.extra},"slime / Miles 的 INT4 QAT；Miles 的 NVFP4 fake-QAT"],
    ["MXFP8 全程（TE 默认）","MXFP8","MXFP8","量化后的函数，外加反向的量化噪声","是（rollout 只用行方向那份）",G("mxfp8"),"Miles --fp8-recipe mxfp8"],
    ["MXFP8 前向 + BF16 反向（high_precision）","MXFP8","BF16，原始值","没量化的 y = x·w，和前向对不上","是",G("mxfp8hp"),"TE NVTE_BACKWARD_OVERRIDE=high_precision"],
    ["NVFP4 前向 + BF16 反向（high_precision）","NVFP4（1D 1×16，按 token）","BF16，原始值","没量化的 y = x·w，和前向对不上","是",G("nvfp4hp"),"Miles 脚本默认；miniTE"],
    ["NVFP4 前向 + BF16 反向（dequantized）","NVFP4（1D 1×16，按 token）","BF16，前向量化值反量化","量化后的函数（= fake quant + STE）","是",G("nvfp4dq"),"The 4-bitter Lesson 的最终配方"],
    ["NVFP4 全程（TE 预训练默认）","NVFP4（权重 16×16）","NVFP4（RHT、随机舍入）","量化后的函数，外加反向的量化噪声","否（训练权重 16×16，rollout 1×16）",G("nvfp4"),"预训练配方"]];
  table($("r-qat"),["做法","前向 GEMM","反向 GEMM","梯度对应的函数","量化结果和 rollout 逐位一致","三遍 GEMM + 量化","相对 BF16","反向要留的（X + W 缓存）","谁在用"],
    rows.map(x=>[esc(x[0]),esc(x[1]),esc(x[2]),esc(x[3]),esc(x[4]),fT(x[5].t),fX(b.t/x[5].t),fB(x[5].extra),esc(x[6])]),[5,6,7]);
  const pk=k=>peakOf(hw,k)/peakOf(hw,"bf16"), ideal=(f,bk)=>3/(1/pk(f)+2/pk(bk));
  const hp=G("nvfp4hp"), hpU=G("nvfp4hp",{tokFused:false}), hpF=G("nvfp4hp",{tokFused:true}), dq=G("nvfp4dq"), tDeq=(dq.q.find(z=>/反量化/.test(z[0]))||[0,0])[1];
  const mem=evOf(/mxfp8-nvfp4-rl/), A=[];
  A.push("只看 GEMM 峰值："+esc(hw.name)+" 上前向 MXFP8 / 反向 BF16 最多 "+V(fX(ideal("mxfp8","bf16")))+"，前向 NVFP4 / 反向 BF16 最多 "+V(fX(ideal("nvfp4","bf16")))+"，全程 NVFP4 是 "+fX(ideal("nvfp4","nvfp4"))+"：反向两遍留在 BF16，前向再快也只占三遍里的一遍。");
  A.push("NVFP4 按 token 的 FP32 scale "+(q.tokFused?"融进 GEMM epilogue 后不多花时间；没融合时":"现在是单独的逐元素 op（先缩放 X，再缩放 Y），")+"三遍从 "+fT(hpF.t)+" 变成 "+V(fT(hpU.t))+"，相对 BF16 从 "+fX(b.t/hpF.t)+" 掉到 "+fX(b.t/hpU.t)+"。"+
    aOut(mem.url,"LMSYS 的低精度 RL 实验")+"里 NVFP4（BF16 反向）训练反而比 BF16 慢，原因就是这个；参数架里的勾选框可以看融合以后的样子。");
  A.push("dequantized 比 high_precision 多一次反量化（X 和 W 回 BF16），这里是 "+V(fT(tDeq))+"，占三遍的 "+pct(tDeq,dq.t)+"；换来的是存给反向的 X 只有 FP4："+fB(dq.saveX)+" 对 "+fB(hp.saveX)+"。"+
    aOut(mem.url,"同一篇")+"测到单个线性层的峰值显存比 TE 默认低精度反向少 70.4%（high_precision）和 49.9%（dequantized），因为不再需要列方向那份量化；本页按副本数算，TE 默认路径里 RHT / 2D 量化的临时 buffer 没建模，所以全程 NVFP4 这一行的显存偏低。");
  A.push("这些都是一个 GEMM 的账；放回整个训练 step、再和 rollout 合成 RL 一步，见<a href=\"#rl/lowprec\">低精度 rollout</a> 第一张表的“训练加速”一列。");
  $("r-qat-notes").innerHTML=ul(A);
  ossTable($("r-qat-oss"),QAT_OSS);
}
