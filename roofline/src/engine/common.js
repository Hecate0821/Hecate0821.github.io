/* 公共层：模型结构、硬件、拓扑、参数推导、roofline 与集合通信的基础函数。推理和训练各自的算子图在 infer.js / train.js。 */
const Common = (function(){

/* 每个元素的字节数（含块 scale）：fp8 是 block FP8 权重，每 128×128 一个 fp32 scale；fp8a 是 block FP8 激活，每 1×128 一个 fp32；
   mxfp8 / mxfp4 每 32 个一个 e8m0；nvfp4 每 16 个一个 e4m3（第二级 fp32 scale 另算） */
const DT = { fp32:4, bf16:2, fp8:1+4/16384, fp8a:1+4/128, mxfp8:1+1/32, mxfp4:0.53125, nvfp4:0.5625 };
const dtB = d => DT[d] ?? 2;
const peakKey = d => d==="fp32"?"bf16":d==="mxfp4"||d==="nvfp4"?"fp4":d==="fp8"||d==="fp8a"||d==="mxfp8"?"fp8":"bf16";

/* 结构字段来自 HF config 与三个引擎 / 两个训练框架的模型文件（交叉核对见“口径与来源”）。 */
const MODELS = {
  qwen35: { name:"Qwen3.5-35B-A3B", short:"Qwen3.5-35B", family:"qwen", D:2048, L:40, V:248320, mtp:1,
    full:{ n:10, kind:"gqa", Hq:16, Hkv:2, dh:256, gate:true },
    lin:{ n:30, kind:"GDN", Hk:16, Hv:32, dk:128, dv:128, conv:4 },
    moe:{ E:256, k:8, F:512, Fs:512, sharedGate:true, Dlat:0 }, dense:{ n:0, F:0 },
    wDefault:"bf16", official:"35B 总 / 3B 激活",
    note:"40 层 = 10 ×（3 × Gated DeltaNet + 1 × 带输出门的 GQA）。每层都是 MoE：256 个 routed expert 选 8，外加 1 个带 sigmoid 门的 shared expert。" },
  kimik3: { name:"Kimi K3", short:"Kimi K3", family:"kimi", D:7168, L:93, V:163840, mtp:0,
    full:{ n:24, kind:"mla", H:96, qLora:1536, kvLora:512, nope:128, rope:64, vh:128, gate:true, nope_only:true },
    lin:{ n:69, kind:"KDA", Hk:96, Hv:96, dk:128, dv:128, conv:4 },
    moe:{ E:896, k:16, F:3072, Fs:6144, sharedGate:false, Dlat:3584 }, dense:{ n:1, F:33792 },
    attnRes:8, wDefault:"mxfp4", official:"2.8T 总 / 104B 激活",
    note:"69 层 KDA + 24 层带 sigmoid 输出门的 MLA（NoPE，但 KV 仍存 512+64 维）。routed expert 在 3584 维 latent 空间里算（router、latent 下投影、shared gate_up 是同一个 bf16 GEMM）；2 个 shared expert 合并成 F=6144，在 7168 维上算（vLLM 每卡常数 SI=768@TP8）。第 0 层是 dense FFN。另有 AttnRes 块残差（最多 8 个快照）。官方 checkpoint 只有 routed expert 是 MXFP4，其余 BF16。按这些字段算出的激活参数比官方的 104B 多几 B，差别来自 shared expert 的口径。" },
  glm53f: { name:"GLM-5.3-Flash", short:"GLM-5.3-Flash", family:"glm", D:4096, L:45, V:154880, mtp:1,
    full:{ n:11, kind:"dsa", H:64, qLora:1536, kvLora:512, nope:256, rope:0, vh:256, gate:false, topk:2048, idxH:32, idxD:128, kpool:4 },
    lin:{ n:34, kind:"KDA", Hk:64, Hv:64, dk:128, dv:128, conv:4 },
    moe:{ E:288, k:8, F:2048, Fs:2048, sharedGate:false, Dlat:0 }, dense:{ n:3, F:12288 },
    mhc:4, wDefault:"fp8", official:"320B 总 / 18B 激活",
    note:"34 层 KDA + 11 层 DSA（i%4==3）。DSA = NoPE MLA（qk_nope = v = 256，KV 512 维）+ KPool lightning indexer：每 4 个 token 池化成一个 key，选 top-512 个池（约 2048 个 token）。前 3 层 dense FFN。4 路 mHC 残差流。主仓库发 FP8 block 权重。" },
  musespark: { name:"Muse Spark Open（816B-A42B）", short:"Muse Spark", family:"muse", D:8192, L:62, V:202048, mtp:0,
    full:{ n:62, kind:"gqa", Hq:128, Hkv:16, dh:64, gate:true, gateHead:true, swa:46, W:2048, gEvery:4, gOff:1 },
    lin:{ n:0, kind:"GDN", Hk:0, Hv:0, dk:128, dv:128, conv:4 },
    moe:{ E:256, k:8, F:4096, Fs:0, sharedGate:false, Dlat:4096 }, dense:{ n:0, F:0 },
    wDefault:"nvfp4", official:"816B 总 / 42B 激活",
    note:"62 层全是 softmax GQA（128 个 q 头、16 个 KV 头、head dim 64，每头一个 sigmoid 门）：46 层滑窗 2048 + RoPE，16 层（i%4==1）看全上下文、不加位置编码，只有这 16 层的 KV 随上下文增长。每层都是 MoE：256 选 8，没有 shared expert；router 在 8192 维上读 fp32，expert 在 4096 维 latent 里算（pre / post_expert_proj 保持 BF16）。官方 NVFP4 checkpoint 只量化 expert（533 GB），BF16 版 1.64 TB。按这些字段算出的激活参数约 40B，官方口径 42B。" },
  qwen38_27b: { name:"Qwen3.8-27B（dense）", short:"Qwen3.8-27B", family:"qwen", D:5120, L:64, V:248320, mtp:1,
    full:{ n:16, kind:"gqa", Hq:24, Hkv:4, dh:256, gate:true },
    lin:{ n:48, kind:"GDN", Hk:16, Hv:48, dk:128, dv:128, conv:4 },
    moe:{ E:0, k:0, F:0, Fs:0, sharedGate:false, Dlat:0 }, dense:{ n:64, F:17408 },
    wDefault:"bf16", official:"27B dense",
    note:"唯一的 dense 对照：64 层 = 16 ×（3 × Gated DeltaNet + 1 × 带输出门的 GQA），和 Qwen3.5-35B-A3B 同一套注意力结构，FFN 换成每层一个 17408 宽的 SwiGLU。config 与 Qwen3.5 / 3.6-27B 完全相同。没有 expert，所以没有 a2a、没有 expert 负载不均，decode 每步要读全部 27B 权重。" }
};

/* 规格表 dense 数值（TFLOP/s、GB/s、GB）；nvl / ib / host 为单 GPU 单向带宽 */
const HW = {
  H100:{ name:"H100 SXM", sm:90, bf16:989, fp8:1979, fp4:1979, hbm:3350, mem:80, nvl:450, ib:50, host:25 },
  H200:{ name:"H200", sm:90, bf16:989, fp8:1979, fp4:1979, hbm:4800, mem:141, nvl:450, ib:50, host:25 },
  B200:{ name:"B200", sm:100, bf16:2250, fp8:4500, fp4:9000, hbm:8000, mem:180, nvl:900, ib:50, host:25 },
  B300:{ name:"B300（HGX）", sm:103, bf16:2250, fp8:4500, fp4:13500, hbm:8000, mem:288, nvl:900, ib:100, host:25 },
  GB200:{ name:"GB200", sm:100, bf16:2500, fp8:5000, fp4:10000, hbm:8000, mem:186, nvl:900, ib:50, host:200 },
  GB300:{ name:"GB300", sm:103, bf16:2500, fp8:5000, fp4:15000, hbm:8000, mem:288, nvl:900, ib:100, host:200 }
};
const TOPO = {
  hgx8:{ name:"8 卡：单台 HGX（NVLink 域 8）", gpus:8, domain:8 },
  nvl64:{ name:"64 卡：同一 NVL72 机柜（NVLink 域 72）", gpus:64, domain:72 },
  hgx64:{ name:"64 卡：8 台 HGX，跨机走 IB", gpus:64, domain:8 },
  nvl128:{ name:"128 卡：2 个 NVL72 机柜，柜间走 IB", gpus:128, domain:72 },
  nvl512:{ name:"512 卡：8 个 NVL72 机柜，柜间走 IB", gpus:512, domain:72 }
};
const peakOf = (hw,d) => hw[peakKey(d)]*1e12;

/* GQA 的 q + 输出门宽度：门按元素（Qwen）是 Hq·dh，按头（Muse Spark）是 Hq */
const qoW = f => f.Hq*f.dh + (f.gate ? (f.gateHead ? f.Hq : f.Hq*f.dh) : 0);
/* 滑窗层每个 query 平均看到的 key 数（整段长 S 的 causal）；全注意力是 S/2 */
const swaKeys = (S, W) => S>W ? W - W*W/(2*S) : S/2;

/* 参数量：按层类型拆开，推理和训练都从这里取 */
function derive(m){
  const D=m.D, f=m.full, l=m.lin, e=m.moe;
  const r = { nMoe: m.L-m.dense.n, De: e.Dlat||D };
  if(f.n>0){
    if(f.kind==="gqa"){
      r.pQKV = D*(qoW(f) + 2*f.Hkv*f.dh); r.pO = f.Hq*f.dh*D; r.pAttnRep = 0;
      r.kvTokElems = 2*f.Hkv*f.dh; r.Hattn=f.Hq; r.dqk=f.dh; r.dv=f.dh;
    } else {
      const dq=f.nope+f.rope;
      r.pAttnRep = D*(f.qLora + f.kvLora + f.rope);                   // fused_qkv_a，各 TP rank 复制
      r.pQKV = f.qLora*f.H*dq + f.kvLora*f.H*(f.nope+f.vh) + (f.gate? D*f.H*f.vh : 0);
      r.pO = f.H*f.vh*D;
      if(f.kind==="dsa") r.pIdx = f.qLora*f.idxH*f.idxD + D*f.idxD + D*f.idxH; else r.pIdx=0;
      r.kvTokElems = f.kvLora + f.rope;                              // K3 NoPE 仍存 64 维槽位（576）；GLM rope=0（512）
      r.Hattn=f.H; r.dqk=dq; r.dv=f.vh;
    }
  } else { r.pQKV=0; r.pO=0; r.pAttnRep=0; r.kvTokElems=0; r.pIdx=0; }
  r.pIdx = r.pIdx||0;
  r.pAttn = r.pQKV + r.pO + r.pAttnRep + r.pIdx;
  if(l.n>0){
    const inW = 2*l.Hk*l.dk + 2*l.Hv*l.dv + (l.kind==="KDA" ? l.Hk*l.dk + 2*l.Hv : 2*l.Hv); // q,k,v,z/gate + β,α (KDA: f 门低秩近似成 H·dk)
    r.linIn = inW; r.pLin = D*inW + l.Hv*l.dv*D;
    r.linState = l.Hv*l.dk*l.dv; r.linConvDim = 2*l.Hk*l.dk + l.Hv*l.dv;
  } else { r.linIn=0; r.pLin=0; r.linState=0; r.linConvDim=0; }
  r.pExpert = 3*r.De*e.F;
  r.pRouted = e.E*r.pExpert;
  r.pShared = 3*D*e.Fs + (e.sharedGate? D : 0);
  r.pRouter = D*e.E;
  r.pLat = e.Dlat ? 2*D*e.Dlat : 0;
  r.pDense = 3*D*m.dense.F;
  r.pEmb = m.V*D; r.pHead = m.V*D;
  r.total = f.n*r.pAttn + l.n*r.pLin + r.nMoe*(r.pRouted+r.pShared+r.pRouter+r.pLat) + m.dense.n*r.pDense + r.pEmb + r.pHead;
  r.active = f.n*r.pAttn + l.n*r.pLin + r.nMoe*(e.k*r.pExpert+r.pShared+r.pRouter+r.pLat) + m.dense.n*r.pDense + r.pHead;
  r.pExpertsAll = r.nMoe*r.pRouted;
  r.pDenseAll = r.total - r.pExpertsAll;
  return r;
}

/* 链路：组内 peer 在同一个 NVLink 域就走 NVLink，否则走 IB（取最慢的那条） */
function link(hw, topo, n, stride){
  if(n<=1) return { bw:Infinity, intra:true, alpha:0 };
  const span = n*(stride||1);
  const intra = span<=topo.domain;
  return { bw:(intra?hw.nvl:hw.ib)*1e9, intra };
}
/* ring all-gather / reduce-scatter：msg 为整份张量字节 */
function agT(msg, n, lk, g){ if(n<=1||msg<=0) return 0; return (lk.intra?g.alphaNv:g.alphaIb)*1e-6*Math.log2(n)+ msg*(n-1)/n/(lk.bw*(g.etaAg||g.etaN)); }
function arT(msg, n, lk, g, algo){
  if(n<=1||msg<=0) return 0; const a=(lk.intra?g.alphaNv:g.alphaIb)*1e-6;
  if(algo==="oneshot") return a + msg*(n-1)/(lk.bw*g.etaN);        // 每个 rank 读其余 n−1 份整消息，受单卡带宽限制
  if(algo==="twoshot") return 2*a + 2*msg*(n-1)/n/(lk.bw*g.etaN);
  return 2*a*Math.log2(n) + 2*msg*(n-1)/n/(lk.bw*g.etaN);         // NCCL ring
}
/* all-to-all：每卡发 sent 字节，(n−1)/n 离开本卡；组跨域时只按最慢链路 */
function a2aT(sent, n, lk, g, alphaUs){ if(n<=1||sent<=0) return 0; return (alphaUs ?? (lk.intra?g.alphaNv:g.alphaIb))*1e-6 + sent*(n-1)/n/(lk.bw*g.etaN); }

/* 单个算子的 roofline：t = max(FLOPs/峰值, 字节/带宽, kernel 数 × 最短 kernel 时间) */
function roof(flops, bytes, peak, hw, g, kernels){
  const tc = flops/(peak*g.etaC), tm = bytes/(hw.hbm*1e9*g.etaM), tf=(kernels||0)*g.kernelUs*1e-6;
  const t = Math.max(tc, tm, tf);
  return { t, tc, tm, tf, bound: t===tf&&tf>0 ? "launch" : tc>=tm ? "compute" : "memory", I: bytes>0?flops/bytes:Infinity };
}

return { DT, dtB, peakKey, peakOf, MODELS, HW, TOPO, derive, qoW, swaKeys, link, agT, arT, a2aT, roof };
})();
