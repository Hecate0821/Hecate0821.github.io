# MoE 训推 Roofline 实验台

浏览器直接打开 `index.html`（`file://` 也能用，不需要构建）。五个页面：

- **总览**：同一个模型的推理一步 decode 和后训练一个 step 并排对比。
- **后训练**：一个 step（fwd / dgrad / wgrad、重算、通信、优化器、显存）；并行（一两层样例里每次通信的顺序和字节，FSDP、CP、EP 一维一维拆开）；低精度训练（只列 Megatron / TE / torchtitan 的默认配方）；和匿名实测对照。
- **推理**：一步 decode / prefill；部署与并行（一层里的通信，TP、DP-attention、EP、PD 分离的取舍）；低精度推理；投机解码（MTP、EAGLE-3、DFlash、DSpark）。
- **RL**：训推不一致；确定性 rollout（batch invariance、可复现、训推逐位一致）；低精度 rollout 和权重同步；低精度训练与 QAT（前向低精度、反向 BF16 的配方）。
- **附录**：公开数据与过去一年的发布；量化基础；模型结构；建模口径和参考的开源实现。

并行和低精度不单独成页，放进训练、推理各自的场景里；模型结构、量化格式这类和场景无关的放附录。拓扑只算两种：单机 8 卡（HGX）和 NVL72 里的 64 卡，所有并行组都在一个 NVLink 域内。地址是 `#页/子页/锚点`，比如 `#infer/spec`、`#train/par/cp`；旧的地址（`#par`、`#quant`、`#public` 等）会跳到新位置。

## 目录

```
index.html            页面骨架；按顺序加载下面的脚本（都是普通 <script>，共享全局作用域）
css/roofline.css
src/engine/           计算模型，不碰 DOM
  common.js           模型 / 硬件 / 拓扑表，derive、链路时间
  par.js              FSDP / CP / EP 的通信与重叠（后训练一步和并行一节共用）；每次通信记成 xc(pass, op, what, bytes, t)
  infer.js            推理一步：SGLang / vLLM / TokenSpeed 的 kernel 序列，含投机解码的 draft、验证、提交
  train.js            训练一步：Megatron-Core / torchtitan；torchtitan 路径调用 Par
  quant.js            低精度 GEMM、一层 MoE、训练三遍 GEMM
src/data/
  db.js               生成文件：数据点库（const DB_RAW）
  query.js            DB.set / DB.where / DB.one / DB.meta
src/ui/               各页的渲染；app.js 最后加载，负责路由和首次渲染
  base.js state.js config.js   格式化与画图底座、全局状态、预设
  overview.js         总览
  train.js par.js calib.js     后训练：一个 step、并行、和匿名实测对照
  comm.js             一层里的通信：后训练并行一节和推理部署一节共用
  infer.js tradeoff.js         推理：一步、部署与并行（三节共用一个参数架）
  quant.js            低精度：推理、训练各一节，附录的格式表
  spec.js             推理 · 投机解码
  rl.js               RL 的四节
  releases.js         附录 · 过去一年的发布；RL 和投机两节的出处也从这里取链接
  public.js dbview.js 附录 · 公开数据、数据点库浏览
  models.js charts.js 模型结构；折线图、甘特图
data/
  anon/*.json         匿名数据点，每个文件一个数据集
  public/*.json       公开数据点（手工整理，带 src / url）：inference、training、quant_evidence、spec、determinism、releases
  db.json             生成文件：规范版本，每个点一行
tools/build_db.py     生成 db.json 和 src/data/db.js
tests/                jsdom 回归测试
```

## 数据点库

```
{ schema: 1,
  sets:   { <id>: { title, origin: "anon" | "public", stage, hw, gpus, topo, model, conditions, dims: [...], metrics: { <名字>: { unit, label } } } },
  points: [ { id, set, ...维度, ...指标 } ] }
```

- **匿名数据点**（`origin: "anon"`）：自己测的数字。只保留测量条件（`conditions`）和数字，不带出处。每个数据集的点有同一组维度和指标，`id` 由维度拼成。
- **公开数据点**（`origin: "public"`）：公开的 PR、博客、论文、发布说明和文档里的数字，点上带 `src` / `url`。量化证据的权重 `w` 在构建时按“时效 × 硬件匹配 × 来源类型”算出，参照日期记在 `sets.quant_evidence.as_of`。
- **发布**（`releases`）：2025-11 到 2026-10 的 LMSYS 博客和 SGLang / vLLM / slime / Miles / AReaL 的 GitHub release，每条是 `releases:<项目>:<版本>`，带日期、链接、要点和标签。页面里引用发布时按项目和版本取链接，库里没有的条目会直接报错。

构建时会检查：匿名点不能带 `src` / `url`，匿名数据集的描述里不能出现出处类的词，id 不重复，没有空数据集。

### 加一个数据点

- 公开数字：在 `data/public/` 对应的 JSON 里加一行（`id`、`date`、`src`、`url`、数字），然后重新构建。
- 匿名实测：在 `data/anon/` 加一个 JSON（标题、测量条件、维度、指标和点），然后重新构建。

```
uv run tools/build_db.py                 # 重新生成 data/db.json 和 src/data/db.js
uv run tools/build_db.py --check         # 只检查生成结果是否最新
uv run tools/build_db.py --sqlite x.sqlite   # 另导出 SQLite（每个数据集一个视图），方便临时查询
```

页面里用 `DB.where("cp_full_model", {model: "qwen35", cp: 8})` 这样的查询取点。

## 测试

```
cd tests && npm install
npm run regression     # 所有模型 × 框架 × 预设跑一遍，再把每一页渲染一遍：报 NaN / undefined 和脚本错误
npm run timing         # 每一页的渲染时间
```
