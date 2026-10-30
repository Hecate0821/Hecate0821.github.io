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
  ["sglang","v0.5.5","DeepSeek 架构的确定性推理；true on-policy 模式对齐 FSDP 训练侧的数值"],
  ["vllm","v0.11.1","batch-invariant 支持 torch.compile，覆盖 Hopper / Blackwell 的 DeepGEMM 和 FlashInfer"],
  ["slime","v0.2.0","重要性采样自定义接口（MIS 等）；R3 与 R2 路由回放；FSDP 后端 dense 模型 true on-policy；确定性 rollout"],
  ["sglang","v0.5.6","RL 权重一致性检查；Qwen3-Next 确定性推理"],
  ["slime","v0.2.1","R3 支持 DP-attention"],
  ["sglang","v0.5.7","返回每个 token 的 routed experts，供训练侧做 R3"],
  ["slime","v0.2.2","R3 完整支持 DeepEP 和 MTP；FSDP 后端的 MIS，修复 TIS / MIS"],
  ["areal","v1.0.0","接 SGLang 返回的 routed_experts，为 MoE routing replay 做准备"],
  ["areal","v1.0.1","完整的 MIS / TIS（此前只有 token 级 MIS）"],
  ["areal","v1.0.4","异步训练统一的 rejection sampling：ratio / KL 指标，token / 序列级，mask / clamp"],
  ["vllm","v0.21.0","routing replay 改用 device cache"],
  ["vllm","v0.22.0","batch invariance 支持 CUTLASS FP8，端到端延迟改善 28.9%"],
  ["lmsys","2026-05-13-no-token-left-behind","Miles 的 TITO：第 n−1 轮的 prompt + response 必须是第 n 轮 prompt 的逐位前缀；重新分词、裁掉历史 reasoning、tool call 重渲染都会破坏它"],
  ["areal","v2.0.0","IcePop、KPop token 掩码并入 rejection sampling"],
  ["slime","v0.3.1","top-p masking，训练复现 rollout 的采样分布；R3 支持 PD 分离"],
  ["areal","v2.1.0","记录每个 token 实际服务的权重版本；rollout 采样确定化"],
  ["slime","v0.3.2","GLM-5 的 Megatron 训练与 SGLang rollout 确定性对齐"],
  ["vllm","v0.29.0","确定性 decode replay；Blackwell 上 batch-invariant kernel 端到端延迟降 33.6%"],
  ["sglang","v0.5.20","return_sampling_mask：返回每步的采样支撑集和 log-prob，训练侧直接回放"],
  ["sglang","v0.5.21","DeepEP v2 支持 batch-invariant 推理"],
  ["miles","v0.1.2","score centering loss：用 sampler 的 top-k 概率修正 off-policy 漂移，可选 TIS / MIS"],
  ["slime","v0.4.0","score centering，支持 top-p replay 和 IS 修正"],
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
  const T=k=>Q.trainGemm(hw,g,k,Mt,N,K,to).t, b=T("bf16");
  const fq=Math.max(g.kernelUs*1e-6, N*K*(2+2)/(hw.hbm*1e9*g.etaM))/Math.max(1,q.nMicro);   // fake quant：读 BF16 权重、写回量化再反量化的 BF16，按 micro-batch 摊
  $("r-qat-lede").innerHTML="一个 GEMM 的三遍（fprop + dgrad + wgrad），形状取低精度训练一节的 N = "+fInt(N)+"、K = "+fInt(K)+"、M = "+fInt(Mt)+"，"+esc(hw.name)+"。显式 fake quant 按每个 micro-batch 读写一遍权重估，反向 STE 直通不多算。";
  const rows=[
    ["BF16 训练 + 低精度 rollout（训练后量化）","BF16","BF16","没量化的 y = x·w","否",b,"多数默认配置"],
    ["显式 fake quant + STE","BF16（权重先量化再反量化）","BF16，STE 直通","量化后的函数","权重逐位一致；激活看 rollout 是否量化",b+fq,"slime / Miles 的 INT4 QAT；Miles 的 NVFP4 fake-QAT"],
    ["NVFP4 前向 + BF16 反向（high_precision）","NVFP4","BF16，原始值","没量化的 y = x·w，和前向对不上","是（1D 1×16，按 token）",T("nvfp4hp"),"Miles 脚本默认；miniTE"],
    ["NVFP4 前向 + BF16 反向（dequantized）","NVFP4","BF16，前向量化值反量化","量化后的函数（= fake quant + STE）","是",T("nvfp4dq"),"The 4-bitter Lesson 的最终配方"],
    ["TE 默认 NVFP4（全程低精度）","NVFP4","NVFP4（RHT、随机舍入）","量化后的函数，外加反向的量化噪声","否（训练权重 16×16，rollout 1×16）",T("nvfp4"),"预训练配方"]];
  table($("r-qat"),["做法","前向 GEMM","反向 GEMM","梯度对应的函数","量化结果和 rollout 逐位一致","三遍 GEMM","相对 BF16","谁在用"],
    rows.map(x=>[esc(x[0]),esc(x[1]),esc(x[2]),esc(x[3]),esc(x[4]),fT(x[5]),fX(b/x[5]),esc(x[6])]),[5,6]);
  ossTable($("r-qat-oss"),QAT_OSS);
}
