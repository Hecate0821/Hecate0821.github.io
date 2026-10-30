/* ---------- 基础配置 ---------- */
const GLOB = { etaC:0.7, etaM:0.8, etaN:0.7, etaAg:0.78, alphaNv:8, alphaIb:20, kernelUs:2, launchUs:4, graphLaunchUs:15, schedUs:0, dpSyncUs:100, a2aLLus:25, deepepUs:10, barrierUs:5 };
const BASE_INF = { engine:"sglang", phase:"decode", N:1, dpA:1, B:64, n:8192, C:8192, P:32768, p:0, wDt:"bf16", eAct:"", dDt:"bf16", kvDt:"bf16", stateDt:"fp32",
  specK:0, specMethod:"mtp", accept:0, graph:true, imb:1.15, redundant:0, dcp:1, prefixCache:true, moePar:"tp", dispFp8:true, tbo:false, hostSyncUs:20,
  memFrac:0.85, reserveGB:6, bi:false, biFactor:2.5, implEff:1, reqPerChunk:1 };
const BASE_TRN = { fw:"megatron", tp:1, cp:1, pp:1, vpp:1, ep:8, etp:1, dispatcher:"alltoall", cpType:"p2p", linCP:"headwise", tok:16384, S:8192, Gtok:2097152,
  prec:"bf16", recompute:"full", rcModules:[], offload:true, overlapOffload:true, cpuAdamGBs:30, overlapGrad:true, overlapParam:true,
  epOverlap:false, sharedOverlap:false, tpOverlap:false, reshard:true, imb:1.15, hostSyncUs:20, workspaceGB:6,
  cpLB:true, hsdp:1, gradDt:"fp32", mokF:0.25, etaA:0 };

const MOEPAR = {
  sglang:[["tp","--moe-a2a-backend none：expert 沿 F 按 TP 切（默认）"],["replicated","--ep-size = --tp-size、不开 a2a：每卡持有 E/N 个完整 expert，输入复制，结果 all-reduce"],["deepep","DeepEP（--moe-a2a-backend deepep）"],["fused","megamoe（a2a 与 expert GEMM 融合）"]],
  vllm:[["tp","不开 EP：expert 按 TP 切"],["ag","EP + allgather_reducescatter（开 EP 后的默认）"],["deepep","EP + DeepEP low-latency / high-throughput"],["fused","EP + flashinfer MNNVL a2a"]],
  tokenspeed:[["replicated","replicated-input EP，结果 all-reduce（默认）"],["tp","expert 按 TP 切"],["deepep","DeepEP（opt-in）"],["fused","flashinfer MNNVL a2a（K3 attention-DP 自动用）"]],
  titan:[["standard","standard all-to-all（k 份拷贝 + D2H 同步）"],["fused","Dist-MoE（对称显存，SM100+）"],["tp","不开 EP"]] };
const MOENAME = { tp:"TP 切 expert", ag:"all-gather + reduce-scatter", replicated:"replicated-input EP + AR", deepep:"DeepEP", fused:"融合 a2a", standard:"torch all-to-all" };
function compatMoe(eng, mp){ const ok=MOEPAR[eng].map(x=>x[0]); if(ok.includes(mp)) return mp;
  if(mp==="replicated"||mp==="ag"||mp==="standard") return Infer.ENGINES[eng].defaultMoe;
  if(mp==="deepep") return eng==="titan"?"standard":"deepep"; return ok[0]; }

const blackwell = () => HW[S.hw].sm>=100;
const hwOK = (h,t) => /^nvl/.test(t) ? /^GB/.test(h) : !/^GB/.test(h);
const prefHW = (mk,t) => /^nvl/.test(t) ? (mk==="kimik3"||mk==="musespark"?"GB300":"GB200") : (mk==="kimik3"?"B300":mk==="glm53f"||mk==="musespark"?"B200":"H200");

/* ---------- 推理预设：引擎 × 模型 × 拓扑 ---------- */
function infPresets(eng, mk, topoKey){
  const big=TOPO[topoKey].gpus>8, G=TOPO[topoKey].gpus, P=[], add=(id,label,src,cfg)=>P.push({id,label,src,cfg});
  const m=MODELS[mk], f8 = m.wDefault==="fp8";
  const X = o => Object.assign({ wDt:m.wDefault, dDt:f8?"fp8":"bf16", kvDt:f8?"fp8":"bf16", stateDt:"fp32" }, o);
  if(eng==="sglang"){
    if(mk==="qwen35"){
      add("rec","TP1，每卡一个实例（"+G+" 个）","SGLang Qwen3.5 文档：H100 / H200 / B200 上 tp=1，多实例由 router 做 DP。",X({N:1,dpA:1,moePar:"tp",B:128}));
      add("mtp","TP1 + MTP（3 个 draft）","--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1。开投机后 max_running_requests 被限到 48。",X({N:1,dpA:1,moePar:"tp",B:48,specK:3,accept:1.8}));
      add("dep","DP-attention 8 + DeepEP 8","--enable-dp-attention --dp 8 --moe-a2a-backend deepep。这个模型没有官方 wide EP 配方，只作对照。",X({N:8,dpA:8,moePar:"deepep",B:64}));
    } else if(mk==="kimik3"){
      if(!big){
        add("rec","TP8（B300 单机）","SGLang Kimi-K3 cookbook：B300 单机 8 卡 TP8；H200 要两台 16 卡 TP16 / EP16。routed expert MXFP4，其余 BF16。",X({N:8,dpA:1,moePar:"tp",B:32}));
        add("dcp","TP8 + DCP8（Balanced）","cookbook 的 Balanced 档加 --dcp-size 8：latent KV 按 8 份切开，单卡能放更多请求。LMSYS day-0 的 decode 就是 TP8 + DCP8。",X({N:8,dpA:1,moePar:"tp",B:96,dcp:8}));
        add("spec","TP8 + DSpark（验证 8 个位置）","LMSYS day-0：DSpark 把 bs=1 decode 从约 113 提到约 423 tok/s；验证 8 个位置时接受长度 chat 约 2.7、few-shot math 约 5.0，这里按 chat。SGLang 开投机后并发限 48。",X({N:8,dpA:1,moePar:"tp",B:32,specK:7,specMethod:"dspark",accept:1.7}));
      } else {
        add("rec","大规模预设：tp=ep=64，attn-dp 8，megamoe","cookbook 大规模预设：--tp-size N --ep-size N --attn-dp-size N/8 --enable-dp-lm-head --moe-a2a-backend megamoe --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16。",X({N:64,dpA:8,moePar:"fused",B:64,kvDt:"fp8",stateDt:"bf16"}));
        add("dp64","attention 全 DP64 + megamoe","同一份 cookbook 的 fully data-parallel 64 卡扫描。",X({N:64,dpA:64,moePar:"fused",B:32,kvDt:"fp8",stateDt:"bf16"}));
        add("tp8","TP8 × 8 个实例","机柜里放 8 个单机形状的实例，对照用。",X({N:8,dpA:1,moePar:"tp",B:32}));
      }
    } else if(mk==="glm53f"){
      if(!big){
        add("rec","TP8，FP8 + FP8 KV","SGLang GLM-5.3-Flash：B200 上 FP8 权重 TP8，trtllm DSA，FP8 KV；H200 高吞吐档 TP8 / EP8 + BF16 KV。",X({N:8,dpA:1,moePar:"tp",B:64}));
        add("mtp","TP8 + MTP","MTP 投机（EAGLE 3 / 1 / 4）。MTP 不支持 DP-attention。",X({N:8,dpA:1,moePar:"tp",B:48,specK:3,accept:1.6}));
        add("dep","DP-attention 8 + DeepEP 8","通用 DP-attention + DeepEP，官方配方里这种形态只出现在 Ascend 上。",X({N:8,dpA:8,moePar:"deepep",B:64}));
      } else {
        add("rec","TP4 × 16 个实例","SGLang / vLLM 的 GB200 配方都是 TP4。",X({N:4,dpA:1,moePar:"tp",B:64}));
        add("dep","DP-attention 64 + DeepEP 64","wide EP 对照，非官方配方。",X({N:64,dpA:64,moePar:"deepep",B:32}));
      }
    } else if(mk==="musespark"){
      if(!big){
        add("rec","TP4 / EP4，NVFP4（B200 / B300）","SGLang Muse Spark Open cookbook 的 B200 / B300 NVFP4 单元：--tp-size 4 --ep-size 4 --quantization modelopt_fp4 --moe-runner-backend flashinfer_trtllm --enable-layernorm-sp --enable-megamoe-for-layernorm-sp，mem 0.8。Low Latency 档另加 DFlash2（8 个 draft token），High Throughput 档去掉投机；这里按 High Throughput。",X({N:4,dpA:1,moePar:"replicated",B:64}));
        add("bf16","TP8 / EP8，BF16","cookbook 的 BF16 单元：--tp-size 8 --ep-size 8，mem 0.85。BF16 权重 1.64 TB，只有 B300（8 × 288 GB）放得下。",X({N:8,dpA:1,moePar:"replicated",B:32,wDt:"bf16"}));
      } else {
        add("rec","TP8 / EP8 跨两个 tray，FP8 KV","cookbook 的 GB300 AgentPerf 部署（Dynamo + SGLang）：NVFP4，TP8 · EP8 跨两个 4 卡 tray 走 MNNVL，--kv-cache-dtype fp8_e4m3，max-running-requests 32，DFlash2 8 个 draft token（这里不算投机）。",X({N:8,dpA:1,moePar:"replicated",B:32,kvDt:"fp8"}));
        add("tp4","TP4 / EP4 × 16 个实例","cookbook 的 GB300 NVFP4 单元：TP4 / EP4。",X({N:4,dpA:1,moePar:"replicated",B:64}));
      }
    } else {
      add("rec","TP1，每卡一个实例，FP8 KV","SGLang Qwen3.8-27B cookbook：所有支持的卡上都单卡跑，每个单元都钉死 --kv-cache-dtype fp8_e4m3；--mamba-full-memory-ratio 要按平均请求长度设，否则 GDN 状态池会限住并发。",X({N:1,dpA:1,moePar:"tp",B:96,kvDt:"fp8"}));
      add("mtp","TP1 + MTP（EAGLE 3 / 1 / 4）","--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4（checkpoint 自带 MTP 头）。开投机后 max_running_requests 被限到 48。",X({N:1,dpA:1,moePar:"tp",B:48,kvDt:"fp8",specK:3,accept:1.8}));
      add("tp2","TP2","两卡一个实例，对照用。",X({N:2,dpA:1,moePar:"tp",B:128,kvDt:"fp8"}));
    }
  } else if(eng==="vllm"){
    if(mk==="qwen35"){
      add("rec","TP1，每卡一个实例","vLLM Qwen3.5 recipe：-tp 1，多实例做 DP。",X({N:1,dpA:1,moePar:"tp",B:128}));
      add("mtp","TP1 + MTP（3 个 token）","--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'。graph 按 token 计，B·4 ≤ 512 才进 graph。",X({N:1,dpA:1,moePar:"tp",B:96,specK:3,accept:1.8}));
      add("dep","DEP8：-dp 8 --enable-expert-parallel","开 EP 后 all2all 后端默认 allgather_reducescatter。",X({N:8,dpA:8,moePar:"ag",B:64}));
    } else if(mk==="kimik3"){
      if(!big){
        add("rec","TP8","vLLM Kimi-K3 recipe：-tp 8，SM100 上 attention 后端 TOKENSPEED_MLA + FP8 KV。",X({N:8,dpA:1,moePar:"tp",B:32,kvDt:"fp8"}));
        add("dep","DEP8","-dp 8 --enable-expert-parallel，默认 allgather_reducescatter。",X({N:8,dpA:8,moePar:"ag",B:32,kvDt:"fp8"}));
      } else {
        add("rec","wide EP：DEP64 + DeepEP","-dp 64 --enable-expert-parallel --all2all-backend deepep_low_latency（同一机柜，对照 vLLM 的 V4.1-Flash wide-EP 博客）。",X({N:64,dpA:64,moePar:"deepep",B:32,kvDt:"fp8"}));
        add("tp8","TP8 × 8 个实例","单机形状放 8 份。",X({N:8,dpA:1,moePar:"tp",B:32,kvDt:"fp8"}));
      }
    } else if(mk==="glm53f"){
      if(!big) add("rec","TP8，FP8","vLLM GLM-5.3-Flash recipe：-tp 8，FP8 权重；SM100 上 sparse MLA 强制 FP8 KV。",X({N:8,dpA:1,moePar:"tp",B:64}));
      else add("rec","TP4 × 16 个实例","GB200 recipe：TP4（另有 TP4 + TP4 的 PD 分离，NIXL 传 KV 和线性状态）。",X({N:4,dpA:1,moePar:"tp",B:64}));
      add("dep",big?"DEP64":"DEP8","-dp N --enable-expert-parallel。",X({N:big?64:8,dpA:big?64:8,moePar:"ag",B:64}));
    } else if(mk==="musespark"){
      add("rec","TEP8：-tp 8 --enable-expert-parallel，NVFP4","vLLM Muse Spark Open recipe 的默认策略 single_node_tep：NVFP4（checkpoint 533 GB）是 NVIDIA 的主路径，一台 8 卡放得下；开 EP 后 all2all 默认 allgather_reducescatter。recipe 默认还开 DSpark（7 个 draft token），这里不算投机。",X({N:8,dpA:1,moePar:"ag",B:64}));
      add("dep","DEP8：-dp 8 --enable-expert-parallel","recipe 列出的 single_node_dep（AMD 不支持）。",X({N:8,dpA:8,moePar:"ag",B:32}));
      if(big) add("tp8","TEP8 × 8 个实例","单机形状放 8 份。",X({N:8,dpA:1,moePar:"ag",B:64}));
    } else {
      add("rec","TP1，每卡一个实例","vLLM Qwen3.8-27B recipe：任何精度都放得进一张 Blackwell（NVFP4 只要 24.6 GiB）；verified 列表有 B200 和 GB300。",X({N:1,dpA:1,moePar:"tp",B:128}));
      add("mtp","TP1 + MTP（3 个 token）","recipe 的默认投机：--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'。",X({N:1,dpA:1,moePar:"tp",B:96,specK:3,accept:1.8}));
    }
  } else if(eng==="tokenspeed"){
    if(mk==="kimik3"){
      add("rec","TP8 / EP8","TokenSpeed Kimi-K3 recipe：TP8 / EP8，MoE 后端 flashinfer_trtllm，FP8 KV，max-num-seqs 32。默认 replicated-input EP：每卡对全部 token 做 router，结果 all-reduce。",X({N:8,dpA:1,moePar:"replicated",B:32,kvDt:"fp8"}));
      add("spec","TP8 / EP8 + 投机","投机时 graph 上限从 160 降到 80；KDA 每步 ≈ 3·S_l 状态流量，外加一次批量 replay 提交。",X({N:8,dpA:1,moePar:"replicated",B:32,kvDt:"fp8",specK:3,accept:1.7}));
      if(big) add("adp","attention-DP 8 + MNNVL a2a","K3 开 attention-DP 时自动走 flashinfer MNNVL a2a。",X({N:8,dpA:8,moePar:"fused",B:32,kvDt:"fp8"}));
    } else if(mk==="glm53f"){
      add("rec","TP8，FP8","TokenSpeed GLM：TP8；GLM 不走 AR 融合，不支持 PD 分离；DSA prefill 走 absorbed 稀疏 MLA，不开 prefill graph。",X({N:8,dpA:1,moePar:"replicated",B:64}));
      add("spec","TP8 + MTP","投机时 graph 上限 80。",X({N:8,dpA:1,moePar:"replicated",B:64,specK:3,accept:1.6}));
    } else if(mk==="musespark"){
      add("rec","TP4 / EP4（套 SGLang 形状）","TokenSpeed 没有 Muse Spark 的 recipe，这里照 SGLang cookbook 的 TP4 / EP4 NVFP4 形状，按 TokenSpeed 默认的 replicated-input EP 算。",X({N:4,dpA:1,moePar:"replicated",B:64}));
    } else {
      const n2 = mk==="qwen38_27b" ? 1 : 2;
      add("rec","TP"+n2+(mk==="qwen38_27b"?"":" / EP"+n2),mk==="qwen38_27b"?"TokenSpeed 没有 Qwen3.8-27B 的 recipe，按 SGLang cookbook 的单卡形状。":"照 TokenSpeed Qwen3 recipe：TP2 + EP2，MoE 后端 flashinfer_cutlass。",X({N:n2,dpA:1,moePar:"replicated",B:128}));
      add("spec","TP"+n2+" + MTP","graph 上限 80（不开投机是 160），sampler 和 draft 都在 graph 里。",X({N:n2,dpA:1,moePar:"replicated",B:80,specK:3,accept:1.8}));
      if(n2>1) add("tp1","TP1","单卡对照。",X({N:1,dpA:1,moePar:"replicated",B:128}));
    }
  } else {
    const T = o => Object.assign({ wDt:"bf16", dDt:"bf16", kvDt:"bf16", stateDt:"fp32", moePar: blackwell()?"fused":"standard", B:64 }, o);
    const st = { qwen35:"仓库标注已支持（Qwen3.5 / 3.6 / 3.8 文本部分）。", qwen38_27b:"Qwen3.5 / 3.6 / 3.8 文本部分已支持；27B dense 没有现成配方，EP 对它不起作用。", musespark:"TitanRL 的大模型 RL 配方就是 Muse Spark Open 816B-A42B（GB300，训推两侧都走 Dist-MoE，8×8 rollout，64K 上下文）。",
      kimik3:"仓库标注 Kimi K3 coming soon，下面只是按 TitanRL 的并行方式套出来的形状。", glm53f:"支持列表里没有 GLM，下面只是按 TitanRL 的并行方式套出来的形状。" }[mk];
    const be = blackwell() ? "EP 走 Dist-MoE。" : "Hopper 上没有 Dist-MoE，EP 走 standard all-to-all。";
    add("rec", big?"816B 配方形状：DP4 × TP4，EP16":"dapo_math 形状：DP4 × TP1，EP4", st+(big?"大模型 RL 配方的 generator 是 16 卡 DP4 / TP4 / EP16。":"Qwen3-30B-A3B 配方的 generator 是 4 卡 DP4 / TP1 / EP4，8 卡上就是 2 个实例。")+be+"TitanRL 同步的是训练侧的 BF16 权重，所以权重和 KV 都是 BF16。",
      T(big?{N:16,dpA:4}:{N:4,dpA:4}));
    add("tp1","TP1 不开 EP","generator 也可以只开 DP：每卡一份完整模型，没有 a2a。",T({N:1,dpA:1,moePar:"tp",B:128}));
  }
  P.push({id:"custom",label:"自定义（改了左侧参数后）",src:"",cfg:null});
  return P;
}

/* ---------- 训练预设：框架 × 模型 × 拓扑 ---------- */
function trnPresets(fw, mk, topoKey){
  const big=TOPO[topoKey].gpus>8, P=[], add=(id,label,src,cfg,need)=>P.push({id,label,src,cfg,need});
  if(fw==="megatron"){
    if(mk==="qwen38_27b"){
      add("miles","miles：TP4，8192 token，CPU Adam","miles scripts/run_qwen3_dense.py 的 Qwen3.8-27B 档：TP4（SP）/ PP1 / CP1 / EP1，max-tokens-per-gpu 8192，full recompute，CPU Adam（offload + overlap D2H/H2D + precision-aware optimizer）。Qwen3.5 / 3.6 / 3.8-27B 结构相同，用同一份参数。",{tp:4,cp:1,ep:1,tok:8192});
    } else if(mk==="musespark"){
      add("miles","miles：TP8 / PP4 / EP8，8192 token，64 卡 GB300","miles scripts/run_muse_spark_32xb300.py：TP8（SP）/ PP4 / CP1 / EP8 / ETP1，首尾 stage 各 15 层，max-tokens-per-gpu 8192，full recompute；Adam 的 FP32 主权重和动量流式放到本地 NVMe（这里按 offload 算）。原配方是 4 台 8 × B300（32 卡），这里只算单机 8 卡和 NVL72 64 卡两种拓扑，所以按一个 NVL72 里的 64 张 GB300 算（DP2）。选它会切到 NVL72。",{tp:8,pp:4,cp:1,ep:8,tok:8192},{topo:"nvl64",hw:"GB300"});
    } else if(mk==="qwen35"){
      add("miles","miles MTP：TP2 / CP2 / EP8，8192 token","miles scripts/run_qwen3_5_35b_a3b_mtp.py：8 卡，TP2 / CP2 / EP8，8192 token，full recompute，CPU Adam。GDN 的 CP 按头 a2a。",{tp:2,cp:2,ep:8,tok:8192});
      add("miles1","miles 另一档：TP1 / EP8","同一文件里注释掉的另一档：1 / 1 / 1 / 8 / 1。",{tp:1,cp:1,ep:8,tok:8192});
    } else if(mk==="glm53f"){
      add("slime","slime：TP1 / PP4 / CP8 / EP16，2048 token","slime scripts/run-glm5.3-flash-300B.sh：64 卡，TP1（不开 SP）/ PP4 / CP8 / EP16，每卡 2048 token，--allgather-cp，CPU Adam。原脚本是 8 台 HGX，这里放进一个 NVL72 的 64 卡（所有组都在 NVLink 内）。选它会切到 NVL72。",{tp:1,pp:4,cp:8,ep:16,tok:2048,cpType:"allgather"},{topo:"nvl64"});
      add("miles","miles：TP8 / PP4 / EP16，GPU Adam","miles scripts/run_glm5_3_flash.py：TP8 / PP4 / CP1 / EP(卡数/4)，micro-batch 1，plain Adam（训练状态 offload 到磁盘，这里按不 offload 算）。按 NVL72 的 64 卡算。",{tp:8,pp:4,cp:1,ep:16,tok:4096,offload:false},{topo:"nvl64"});
    } else if(mk==="kimik3"){
      add("miles","miles：TP4 / PP8 / CP2 / EP8，64 卡 GB300","miles scripts/run_kimi_k3.py（文档里的命令是 LoRA）：64 卡（16×4 GB300），TP4 / PP8 / CP2 / EP8，8192 token，CPU Adam。这里按全参数算，显存几乎贴着上限。选它会切到 NVL72。",{tp:4,pp:8,cp:2,ep:8,tok:8192},{topo:"nvl64",hw:"GB300"});
    }
    const dense=!MODELS[mk].moe.E;
    add("generic",dense?"通用：TP2":big?"通用：TP4 / CP2 / EP16":"通用：TP2 / EP8","没有配方时的起点。",dense?{tp:2,cp:1,ep:1,tok:8192}:big?{tp:4,cp:2,ep:16,tok:8192}:{tp:2,cp:1,ep:8,tok:8192});
  } else {
    const bw = blackwell(), disp = bw?"distmoe":"standard";
    const st = { kimik3:"仓库标注 Kimi K3 coming soon。", glm53f:"GLM 不在 torchtitan 的模型列表里，只是套形状。", musespark:"这个配方本身就是 Muse Spark Open 816B-A42B 的。", qwen38_27b:"dense 模型没有 expert，EP 不起作用，只剩 FSDP。" }[mk]||"";
    if(!big) add("dapo","TitanRL dapo_math 形状：FSDP + EP4","torchtitan RL 的 Qwen3-30B-A3B 配方：trainer 4 卡 FSDP4 / EP4，这里放到 8 卡（dense 参数 FSDP8，expert 在 2 卡上 FSDP）。默认 AC 只重算 w13，不重放 a2a。"+(bw?"Blackwell 上 EP 走 Dist-MoE。":"Hopper 上没有 Dist-MoE。")+st,
      {tp:1,cp:1,ep:MODELS[mk].moe.E?4:1,tok:8192,dispatcher:disp,cpType:"allgather",linCP:"state",recompute:"default",offload:false});
    else add("big","TitanRL 大模型配方形状：TP4 / CP4 / EP16","torchtitan RL 大模型配方（GB300）的 trainer：TP4、Ulysses CP4（varlen）、EP16，expert 在剩下的卡上 FSDP。"+st,
      {tp:4,cp:4,ep:16,tok:16384,dispatcher:disp,cpType:"ulysses",linCP:"ulysses",recompute:"default",offload:false});
    if(MODELS[mk].moe.E) add("ep8","FSDP + EP8，全重算","EP8，full AC。",{tp:1,cp:1,ep:8,tok:16384,dispatcher:disp,cpType:"allgather",linCP:"state",recompute:"full",offload:false});
    if(mk==="qwen35") add("h200cp8","8 × H200 实测：CP8 AllGather / EP4，256K","匿名实测的设置：8 × H200，一篇 256K 文档切 8 份（每卡 32K），CP8 AllGather（连续切分、不做 causal 均衡），EP4、expert FSDP 2，full AC。“并行”页最后一张表拿它和实测对照。",
      {tp:1,cp:8,ep:4,tok:32768,S:262144,Gtok:262144,dispatcher:"standard",cpType:"allgather",cpLB:false,linCP:"state",recompute:"full",offload:false},{topo:"hgx8",hw:"H200"});
    if(mk==="qwen38_27b") add("uly256","Ulysses CP8，256K","dense 27B 的长上下文：一篇 256K 切 8 份，Ulysses CP8，dense 参数在 8 卡上 FSDP。",
      {tp:1,cp:8,ep:1,tok:32768,S:262144,Gtok:262144,cpType:"ulysses",linCP:"ulysses",recompute:"full",offload:false});
    if(mk==="kimik3") add("mok64","GB300 × 64：EP64 MoK，MXFP8","Cursor 训练 K2.7 形状模型的设置套到 K3：EP64 走 Mixture-of-Kittens，MXFP8 GEMM，CP4。原设置是 8 个 NVL72 机柜，这里只算一个机柜的 64 卡；K3 有 2.8T 参数，Adam 放 CPU。",
      {tp:1,cp:4,ep:64,tok:16384,S:65536,Gtok:2097152,dispatcher:"mok",cpType:"allgather",linCP:"state",prec:"mxfp8",recompute:"full",offload:true},{topo:"nvl64",hw:"GB300"});
    if(mk==="musespark") add("dm64","GB300 × 64：EP16 Dist-MoE，CP4","一个 NVL72 机柜，EP16 走 Dist-MoE，dense 参数在 64 卡上 FSDP，全部在 NVLink 内。",
      {tp:1,cp:4,ep:16,tok:16384,S:65536,Gtok:4194304,dispatcher:"distmoe",cpType:"allgather",linCP:"state",recompute:"full",offload:false},{topo:"nvl64",hw:"GB300"});
  }
  P.push({id:"custom",label:"自定义",src:"",cfg:null});
  return P;
}
