# LLM 系统原理学习与代码实践计划

日期：2026-08-24
状态：系统原理主线已确认，待文档审阅

## 1. 适用背景

本计划面向具备以下经验的工程师：

- 有数年分布式系统开发经验；
- 做过机器学习平台、训练平台或模型服务平台；
- 熟悉 Python、Linux、容器、监控和常见服务治理；
- 可在 AutoDL 使用一张 32GB 显存 GPU；
- 可额外短租 2～4 张 GPU 完成 1～2 天多卡实验。

因此，本计划不重复讲解 Python、Docker、Kubernetes、HTTP API 和基础监控，而把时间集中在：

1. Transformer 的实际计算过程；
2. 训练阶段的数值、显存和数据问题；
3. SFT、QLoRA、DPO 的作用边界；
4. LLM 推理调度、KV Cache 和量化；
5. FSDP、ZeRO、Tensor Parallel 等模型分布式机制；
6. 质量、性能和成本的联合评测。

## 2. 学习结果

完成计划后，应能够基于代码、日志和测量结果回答：

- 一次 Transformer 前向计算具体执行了什么；
- 模型参数、梯度、优化器状态、激活值和 KV Cache 分别占用多少显存；
- 为什么训练发生 OOM、Loss 异常、梯度爆炸或吞吐下降；
- LoRA/QLoRA 修改了哪些参数，节省了什么，没有节省什么；
- Prefill 和 Decode 为什么具有不同的计算特征；
- vLLM 的 PagedAttention 和 Continuous Batching 解决了什么问题；
- BF16、INT8、INT4 如何影响显存、吞吐和质量；
- DDP、FSDP/ZeRO、TP、PP 和 EP 分别适合什么场景；
- 如何用可复现实验判断模型或系统优化是否有效。

这不是“大模型专家速成计划”。12 个全职学习日可以建立完整认知框架并获得独立实验能力，
但 CUDA Kernel、超大规模预训练、多节点容错和生产级集群调优仍需要后续项目经验。

## 3. 范围与非目标

### 3.1 核心范围

- 从零实现并训练一个 50M～150M 参数的 Decoder-only Transformer；
- 使用 7B/8B 开源模型完成 QLoRA SFT；
- 使用小规模偏好数据完成一次 DPO；
- 建立独立于训练 Loss 的质量评测；
- 使用 Transformers 和 vLLM 分别部署并对比；
- 完成 BF16、INT8、INT4 推理对比；
- 完成至少一次真实多 GPU 训练或推理实验；
- 实现 OpenAI 兼容服务、压测、监控和故障实验。

### 3.2 非目标

- 从零预训练 7B 或更大模型；
- 单卡完成 7B 全参数微调；
- 把 RAG、Agent 或 Prompt Engineering 作为主线；
- 大量学习 LangChain 等应用编排框架；
- 追求某个公开榜单的最高分；
- 在学习阶段过早进行复杂平台封装。

## 4. 固定实验配置

为了减少无意义的变量，整个计划使用一套固定基线。

### 4.1 模型

- 原理实验：自建 50M～150M Decoder-only Transformer；
- 小语料：TinyStories 或等量的已清洗文本；
- 微调主模型：Qwen2.5-7B-Instruct，或同规模、架构文档完整的 7B/8B 模型；
- DPO 模型：优先复用主模型；若 32GB 显存无法在无隐式 CPU Offload 下稳定运行，则降至
  同系列 1.5B/3B 模型，并在报告中明确实验边界；
- 多卡训练对照：使用 DDP 和 FSDP/ZeRO 都能稳定容纳的约 1B 模型，保证实验可比；
- Tensor Parallel 推理：继续使用 7B/8B 主模型；
- 主任务：选择一个有明确输入输出标准的领域问答或信息抽取任务；
- 默认最大训练序列长度：2048；
- 长上下文仅用于推理侧对比，不作为首轮训练目标。

选择 7B/8B 而不是更大模型，是为了在 32GB 显存内保留足够的实验空间。学习目标是比较机制，
不是证明单卡可以勉强加载多大的模型。

### 4.2 软件环境

训练和服务使用两个环境，避免 vLLM、PyTorch、CUDA 和训练依赖互相污染：

```text
llm-train   # PyTorch、Transformers、Datasets、PEFT、TRL
llm-serve   # vLLM、FastAPI、压测和监控依赖
```

开始前先记录 AutoDL 镜像和驱动，不盲目升级预装 PyTorch：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory)"
```

核心依赖：

```text
torch
transformers
datasets
accelerate
peft
trl
bitsandbytes
sentencepiece
tensorboard
pytest
vllm
fastapi
uvicorn
prometheus-client
httpx
```

具体版本应在创建环境当天根据 AutoDL 的 CUDA/PyTorch 组合锁定到
`requirements-train.lock` 和 `requirements-serve.lock`，不要在计划文档里假设长期有效的版本号。

## 5. 实验仓库结构

学习代码单独建立仓库，不与现有 GIS 项目代码混合：

```text
llm-systems-lab/
├── README.md
├── Makefile
├── requirements-train.lock
├── requirements-serve.lock
├── configs/
│   ├── tiny_pretrain.yaml
│   ├── qlora_sft.yaml
│   ├── dpo.yaml
│   └── serve.yaml
├── src/llm_lab/
│   ├── data/
│   │   ├── tokenizer.py
│   │   ├── pretrain_dataset.py
│   │   └── sft_dataset.py
│   ├── model/
│   │   ├── attention.py
│   │   ├── blocks.py
│   │   ├── minillm.py
│   │   └── generation.py
│   ├── training/
│   │   ├── pretrain.py
│   │   ├── sft_qlora.py
│   │   ├── dpo.py
│   │   └── checkpoint.py
│   ├── evaluation/
│   │   ├── quality.py
│   │   └── regression.py
│   ├── inference/
│   │   ├── hf_server.py
│   │   ├── vllm_client.py
│   │   └── benchmark.py
│   └── distributed/
│       ├── fsdp_train.py
│       └── communication_probe.py
├── tests/
├── scripts/
├── reports/
│   ├── environment.md
│   ├── tiny-pretrain.md
│   ├── qlora-sft.md
│   ├── quality-eval.md
│   ├── inference-benchmark.md
│   ├── distributed-comparison.md
│   └── final-report.md
└── artifacts/
    └── README.md
```

模型权重、原始数据和大体积日志不得提交 Git。`artifacts/README.md` 只记录对象存储路径、
校验和、生成命令和实验 ID。

## 6. 全局工程规则

所有练习代码都遵守以下检查项：

- [ ] 所有实验使用 YAML 配置，不把关键超参数散落在命令行和代码中；
- [ ] 固定并记录 Python、NumPy、PyTorch 和 CUDA 随机种子；
- [ ] 每次实验生成唯一 `run_id`；
- [ ] 保存代码 commit、配置快照、环境版本和数据版本；
- [ ] 训练支持从 Checkpoint 恢复；
- [ ] 记录 Loss、学习率、梯度范数、tokens/s 和峰值显存；
- [ ] 推理记录 TTFT、TPOT、端到端延迟、tokens/s、并发和错误率；
- [ ] 不以“脚本退出码为 0”作为实验成功标准；
- [ ] 每项优化必须有基线、单变量变更和对比结论；
- [ ] 核心张量操作必须有 Shape、因果 Mask 和数值正确性测试。

## 7. 12 个全职学习日

每天按 6～8 小时设计。建议分配为理论 1～1.5 小时、编码 4～5 小时、实验与复盘 1～2 小时。

### 第 1 天：环境基线与 Transformer 计算地图

**目标**

- 固定可复现实验环境；
- 建立从 Token 到 Logits 的完整计算地图；
- 能够估算模型权重和训练状态的显存。

**理论**

- Decoder-only Transformer 数据流；
- Embedding、Attention、MLP、Residual、RMSNorm、LM Head；
- 参数量、FLOPs 和显存的基本关系；
- BF16、FP16、FP32 的用途差异。

**代码实践**

- [ ] 创建独立训练环境并保存环境探针输出；
- [ ] 实现 `scripts/check_environment.py`；
- [ ] 实现 `scripts/estimate_memory.py`；
- [ ] 输入层数、hidden size、head 数、词表和精度，估算参数量与权重显存；
- [ ] 增加梯度、AdamW 状态和激活值的粗略估算；
- [ ] 为 7B 模型计算 BF16、INT8、INT4 权重下限；
- [ ] 在 `reports/environment.md` 记录 GPU、驱动、CUDA、PyTorch 和磁盘信息。

**验收**

- 能解释为什么“7B BF16 权重约 14GB”不代表 16GB GPU 就能稳定训练；
- 估算结果与实际加载模型的峰值显存误差控制在可解释范围；
- 环境可以在新终端中一条命令重新激活。

### 第 2 天：手写 Attention、RoPE 和因果 Mask

**目标**

- 不依赖 Transformers 的 Attention 实现完成前向计算；
- 理解 MHA、GQA、MQA 的 Shape 和 KV Cache 差异。

**理论**

- Q、K、V 投影和 scaled dot-product attention；
- Causal Mask；
- RoPE；
- MHA、GQA、MQA；
- PyTorch SDPA 与 FlashAttention 的接口边界。

**代码实践**

- [ ] 实现 `model/attention.py`；
- [ ] 实现标准 MHA；
- [ ] 实现 RoPE，并测试不同位置的旋转结果；
- [ ] 实现 Causal Mask；
- [ ] 扩展到 GQA；
- [ ] 与 PyTorch `scaled_dot_product_attention` 输出对比；
- [ ] 编写 Shape、Mask 和数值误差测试；
- [ ] 用 profiler 比较手写版本和 SDPA 的速度、显存。

**验收**

- 未来 Token 不影响当前 Token 的输出；
- 在相同输入和权重下，手写实现与参考实现误差满足所用精度要求；
- 能根据 `num_attention_heads` 和 `num_key_value_heads` 计算 KV Cache 大小。

### 第 3 天：完成最小 Decoder-only Transformer

**目标**

- 组装可训练的最小语言模型；
- 验证前向、反向和自回归生成。

**理论**

- Pre-Norm、RMSNorm、SwiGLU；
- Weight Tying；
- Cross Entropy 和 next-token prediction；
- 初始化方式和残差路径。

**代码实践**

- [ ] 实现 `model/blocks.py`；
- [ ] 实现 RMSNorm、SwiGLU、Decoder Block；
- [ ] 实现 `model/minillm.py`；
- [ ] 实现 shifted labels 和 LM Loss；
- [ ] 实现 greedy、temperature、top-k、top-p 生成；
- [ ] 检查参数量与配置推导是否一致；
- [ ] 对一个极小 batch 执行 forward、backward 和 optimizer step；
- [ ] 增加过拟合单 batch 的测试。

**验收**

- 模型可在单 batch 上快速过拟合；
- 关闭 Dropout 后，同输入和权重产生确定性输出；
- 每个核心张量的 Shape 都能在纸上推导。

### 第 4 天：小模型预训练闭环

**目标**

- 从原始文本跑通 Tokenizer、Dataset、训练、保存和生成；
- 观察真实训练曲线，而不是只验证代码可执行。

**理论**

- BPE/SentencePiece；
- 文本切分、EOS、Packing、Document Boundary；
- Token 数、Batch Token 和训练步数；
- Warmup、Cosine Schedule、Gradient Clipping。

**代码实践**

- [ ] 训练或固定一个小词表 Tokenizer；
- [ ] 实现流式文本 Tokenization 和定长 Packing；
- [ ] 实现 `training/pretrain.py`；
- [ ] 支持梯度累积、混合精度和梯度裁剪；
- [ ] 每 N 步保存 Checkpoint 和样例生成结果；
- [ ] 从 Checkpoint 恢复并验证 step、optimizer、scheduler 连续；
- [ ] 至少完成一次 50M～150M 模型训练；
- [ ] 写 `reports/tiny-pretrain.md`。

**验收**

- Train/Validation Loss 有明确下降；
- 恢复训练后的曲线没有非预期跳变；
- 固定 Prompt 的生成结果随训练进度改善；
- 能区分数据管线瓶颈和 GPU 计算瓶颈。

### 第 5 天：训练显存、吞吐和数值稳定性

**目标**

- 用证据解释训练资源消耗；
- 比较 Gradient Checkpointing、精度和 Batch 策略。

**理论**

- 权重、梯度、优化器状态和激活值；
- Gradient Accumulation；
- Activation Checkpointing；
- FP16 Loss Scaling 与 BF16；
- NaN、Inf、梯度爆炸和数据异常。

**代码实践**

- [ ] 为训练循环增加峰值显存和 tokens/s 统计；
- [ ] 比较 BF16 与 FP32；
- [ ] 比较开启和关闭 Gradient Checkpointing；
- [ ] 比较不同 micro-batch 与 accumulation 组合；
- [ ] 记录梯度范数和非有限值；
- [ ] 人工注入异常 batch，验证日志能定位问题；
- [ ] 生成显存与吞吐对照表。

**验收**

- 所有比较只改变一个核心变量；
- 能解释 Checkpointing 为什么省显存但增加计算；
- 能根据 OOM 日志判断问题更可能来自权重、激活值还是临时张量。

### 第 6 天：SFT 数据与 QLoRA

**目标**

- 构造可审计的 SFT 数据；
- 在 32GB 单卡启动 7B/8B QLoRA。

**理论**

- Base Model、Instruct Model 和 Chat Template；
- LoRA 的低秩更新；
- NF4、Double Quantization 和计算精度；
- Target Modules；
- Label Mask 与只训练 Assistant Token。

**代码实践**

- [ ] 定义统一的训练样本 Schema；
- [ ] 实现格式校验、去重、长度分布和泄漏检查；
- [ ] 正确应用模型 Chat Template；
- [ ] 实现 Assistant Token Label Mask；
- [ ] 实现 `training/sft_qlora.py`；
- [ ] 打印可训练参数及占比；
- [ ] 保存训练前固定评测集输出；
- [ ] 启动首轮 QLoRA 并记录峰值显存。

**验收**

- 随机抽查 Token、Label 和 Mask，确认系统/用户 Token 未误算 Loss；
- 可训练参数仅为预期 LoRA 参数；
- 训练数据和评测数据不存在直接重复；
- 32GB 显存内训练稳定，没有依赖模糊的自动降级。

### 第 7 天：SFT 评测与错误分析

**目标**

- 判断微调是否真正改善目标任务；
- 把训练 Loss 与最终质量分离。

**理论**

- Exact Match、F1、结构化输出合法率；
- 生成参数对评测的影响；
- LLM-as-a-Judge 的偏差；
- 数据污染、过拟合和灾难性遗忘。

**代码实践**

- [ ] 实现 `evaluation/quality.py`；
- [ ] 固定 Prompt、采样参数、随机种子和最大输出长度；
- [ ] 比较原模型与 LoRA 模型；
- [ ] 按题型和长度分桶；
- [ ] 保存原始输入、原始输出、解析结果和评分；
- [ ] 建立 20～50 条人工回归集；
- [ ] 对失败样本分类，而不是只报告平均分；
- [ ] 写 `reports/quality-eval.md`。

**验收**

- 报告同时包含改善、退化和无变化样本；
- 任一汇总分都能追溯到原始输出；
- 能判断问题来自数据、Prompt、解码参数还是模型能力。

### 第 8 天：偏好优化与 DPO 边界

**目标**

- 跑通一次小规模 DPO；
- 理解 DPO 改变行为偏好而不是注入可靠知识。

**理论**

- Chosen/Rejected 数据；
- Reference Model；
- DPO Loss 和 Beta；
- Reward Hacking、长度偏好和格式偏差。

**代码实践**

- [ ] 构造小规模 preference dataset；
- [ ] 校验 chosen/rejected 的 Prompt 一致；
- [ ] 实现或配置 `training/dpo.py`；
- [ ] 优先使用 Adapter 禁用方式提供 Reference 行为，避免复制完整 7B 模型；
- [ ] 如果显存仍不足，按固定规则降至 1.5B/3B，并记录原因，禁止静默启用 CPU Offload；
- [ ] 记录 chosen/rejected reward、margin 和长度；
- [ ] 比较 SFT 与 DPO 模型；
- [ ] 检查是否仅学习到更长、更模板化的回答；
- [ ] 记录 DPO 无明显收益或造成退化的证据。

**验收**

- 能解释 SFT 与 DPO 分别解决什么问题；
- 不使用训练 Loss 单独证明对齐有效；
- 偏好数据中的系统性偏差被明确记录。

### 第 9 天：自回归推理与 KV Cache

**目标**

- 手写带 KV Cache 的生成循环；
- 理解 Prefill 与 Decode 的不同瓶颈。

**理论**

- Autoregressive Decode；
- Prefill、Decode、TTFT、TPOT；
- KV Cache 的布局和生命周期；
- Memory-bound 与 Compute-bound；
- 长上下文对显存和延迟的影响。

**代码实践**

- [ ] 为最小模型实现无 Cache 生成；
- [ ] 实现带 KV Cache 的增量生成；
- [ ] 验证两种生成在 greedy 模式下输出一致；
- [ ] 测量不同输入长度和输出长度的耗时；
- [ ] 实现 KV Cache 估算脚本；
- [ ] 对 7B/8B 模型测量 Prefill 和 Decode；
- [ ] 绘制上下文长度与显存、TTFT 的关系。

**验收**

- 能解释为什么 Decode 每一步只新增一个 Token，但仍可能很慢；
- KV Cache 公式与实测趋势一致；
- 不混用 TTFT、TPOT、端到端延迟和吞吐量。

### 第 10 天：vLLM、连续批处理与量化

**目标**

- 对比 Transformers 和 vLLM；
- 用同一评测集衡量量化收益和损失。

**理论**

- Static Batching 与 Continuous Batching；
- PagedAttention；
- 请求调度、抢占和 Prefix Cache；
- Weight-only Quantization；
- 量化对权重、计算和 KV Cache 的不同影响。

**代码实践**

- [ ] 启动 Transformers 基线服务；
- [ ] 启动 vLLM OpenAI 兼容服务；
- [ ] 实现统一客户端和负载生成器；
- [ ] 比较并发 1、8、32、64；
- [ ] 比较短输入、长输入和长短混合请求；
- [ ] 根据当前 GPU 和推理引擎的支持矩阵选择明确的 INT8/INT4 格式；
- [ ] 比较 BF16、INT8、INT4；后端不支持的组合标记为不适用，不通过更换引擎伪装同口径对比；
- [ ] 同时运行质量回归集；
- [ ] 写 `reports/inference-benchmark.md`。

**验收**

- 报告包含 TTFT、TPOT、P50/P95/P99、输出 tokens/s、错误率和峰值显存；
- 量化结论同时考虑性能和质量；
- 能解释低并发下 vLLM 不一定表现出最大优势的原因。

### 第 11 天：真实多 GPU 并行实验

**目标**

- 把已有分布式经验映射到模型训练和推理；
- 使用真实通信数据比较并行策略。

**理论**

- DDP；
- ZeRO-1/2/3 与 FSDP Full Shard；
- Tensor Parallel；
- Pipeline Parallel；
- Expert Parallel；
- 参数、梯度、优化器状态、激活值和 KV Cache 的切分方式。

**代码实践**

- [ ] 短租 2～4 张同型号 GPU；
- [ ] 编写 `distributed/communication_probe.py` 测量集合通信；
- [ ] 使用约 1B 模型和全参数训练，以 DDP 跑一组基线；
- [ ] 使用同一模型、数据、有效 batch tokens 和训练步数，以 FSDP 或 ZeRO-3 跑对照实验；
- [ ] 记录每卡显存、step time、通信占比和扩展效率；
- [ ] 使用 7B/8B 主模型以 Tensor Parallel 部署一次推理服务；
- [ ] 人工结束一个 Worker，记录失败表现和恢复边界；
- [ ] 写 `reports/distributed-comparison.md`。

**验收**

- 能说明 FSDP/ZeRO 与 TP 解决的不是同一个问题；
- 所有对比使用相同有效 batch tokens；
- 能从 trace 中指出主要 collective 和等待位置；
- 不用单次运行结果宣称线性扩展。

### 第 12 天：服务闭环、故障实验与最终复现

**目标**

- 把训练、评测和推理产物串成可复现闭环；
- 验证容量边界和故障行为；
- 形成下一阶段学习缺口清单。

**理论**

- Admission Control；
- Backpressure；
- 动态批处理与 SLO；
- 模型版本、灰度和回滚；
- 质量回归与性能回归的发布门禁。

**代码实践**

- [ ] 提供 OpenAI 兼容 API；
- [ ] 增加健康检查、就绪检查和 Prometheus 指标；
- [ ] 增加并发上限、队列上限、请求超时和取消；
- [ ] 注入超长 Prompt、客户端中断、OOM 和 Worker 退出；
- [ ] 验证错误码、日志字段和指标是否可定位问题；
- [ ] 从干净环境执行一次端到端复现；
- [ ] 整理所有配置、脚本、报告和权重索引；
- [ ] 完成 `reports/final-report.md`。

**验收**

- 一条命令可启动服务，一条命令可执行回归和压测；
- 服务在超载时明确拒绝或排队，不发生无界资源增长；
- 最终报告明确区分事实、实验结果和推测；
- 形成后续 30 天深入方向，而不是继续横向堆框架。

## 8. 四周业余版：28 天每日安排

每天按 2～3 小时设计。该版本覆盖相同实验，但拆分为更小的编码单元。每周第 7 天用于复盘、
补测或处理 AutoDL 环境问题，不额外引入新主题。

### 第 1 周：模型计算与最小实现

**第 1 天：环境与基线**

- [ ] 创建训练环境；
- [ ] 运行 GPU/CUDA/PyTorch 探针；
- [ ] 锁定依赖；
- [ ] 创建仓库结构和实验记录模板。

产物：`reports/environment.md`。

**第 2 天：Attention 数学与 Shape**

- [ ] 手工推导 Q/K/V、Attention Scores 和输出 Shape；
- [ ] 实现单头 Attention；
- [ ] 添加 Causal Mask 测试。

产物：单头 Attention 与单元测试。

**第 3 天：MHA、GQA 与 RoPE**

- [ ] 实现多头拆分与合并；
- [ ] 实现 RoPE；
- [ ] 扩展到 GQA；
- [ ] 与 SDPA 对比。

产物：`attention.py` 和数值对照结果。

**第 4 天：Transformer Block**

- [ ] 实现 RMSNorm；
- [ ] 实现 SwiGLU；
- [ ] 组装 Pre-Norm Decoder Block。

产物：`blocks.py`。

**第 5 天：最小语言模型**

- [ ] 实现 Embedding、Block Stack、LM Head；
- [ ] 实现 LM Loss；
- [ ] 完成 forward/backward 测试。

产物：`minillm.py`。

**第 6 天：生成与 Tokenizer**

- [ ] 实现 greedy、temperature、top-k、top-p；
- [ ] 准备小语料和 Tokenizer；
- [ ] 检查 EOS 和文档边界。

产物：`generation.py`、Tokenizer 配置。

**第 7 天：周复盘**

- [ ] 过拟合单 batch；
- [ ] 修复 Shape、Mask 和非确定性问题；
- [ ] 写出从 Token 到 Loss 的完整数据流；
- [ ] 清理无法解释的代码。

产物：第一周复盘记录。

### 第 2 周：预训练、显存与 QLoRA

**第 8 天：预训练数据管线**

- [ ] 实现流式 Tokenization；
- [ ] 实现定长 Packing；
- [ ] 统计 Token 数和长度分布。

产物：`pretrain_dataset.py`。

**第 9 天：训练循环**

- [ ] 实现 Optimizer、Scheduler、Gradient Clipping；
- [ ] 记录 Loss、学习率、梯度范数和 tokens/s；
- [ ] 启动小模型训练。

产物：`pretrain.py` 和首轮日志。

**第 10 天：Checkpoint 与恢复**

- [ ] 保存模型、优化器、Scheduler、RNG 和 step；
- [ ] 中断并恢复训练；
- [ ] 对比恢复前后曲线。

产物：Checkpoint 恢复测试。

**第 11 天：显存与吞吐实验**

- [ ] 比较 BF16/FP32；
- [ ] 比较 Checkpointing 开关；
- [ ] 比较 micro-batch 和 accumulation；
- [ ] 记录峰值显存。

产物：显存/吞吐矩阵。

**第 12 天：SFT 数据**

- [ ] 定义 Schema；
- [ ] 去重、分割并检查泄漏；
- [ ] 应用 Chat Template；
- [ ] 检查 Label Mask。

产物：`sft_dataset.py` 和数据质量报告。

**第 13 天：QLoRA 启动**

- [ ] 加载 4-bit 模型；
- [ ] 配置 LoRA Target Modules；
- [ ] 打印可训练参数；
- [ ] 启动首轮训练。

产物：`sft_qlora.py` 和运行配置。

**第 14 天：周复盘**

- [ ] 检查训练曲线和异常样本；
- [ ] 保存 Adapter；
- [ ] 生成训练前后固定样例；
- [ ] 补齐预训练和 QLoRA 报告。

产物：`tiny-pretrain.md`、`qlora-sft.md`。

### 第 3 周：质量评测与推理系统

**第 15 天：评测框架**

- [ ] 固定评测集和生成参数；
- [ ] 实现结构化评分；
- [ ] 保存所有原始输出。

产物：`quality.py`。

**第 16 天：SFT 错误分析**

- [ ] 对比原模型和 Adapter；
- [ ] 按类型、长度分桶；
- [ ] 标记改善、退化和无变化样本。

产物：`quality-eval.md`。

**第 17 天：偏好数据**

- [ ] 构造 chosen/rejected；
- [ ] 检查长度和格式偏差；
- [ ] 建立 DPO 前基线。

产物：偏好数据质量报告。

**第 18 天：DPO**

- [ ] 运行小规模 DPO；
- [ ] 记录 reward、margin 和长度；
- [ ] 比较 SFT 与 DPO 输出。

产物：`dpo.py` 和对比结论。

**第 19 天：KV Cache**

- [ ] 实现无 Cache 和有 Cache 生成；
- [ ] 验证 greedy 输出一致；
- [ ] 估算和测量 KV Cache。

产物：KV Cache 测量脚本。

**第 20 天：vLLM 服务**

- [ ] 启动 Transformers 基线；
- [ ] 启动 vLLM；
- [ ] 用统一客户端验证流式与非流式输出。

产物：两个可调用服务和客户端。

**第 21 天：周复盘与基础压测**

- [ ] 测量并发 1、8、32；
- [ ] 记录 TTFT、TPOT、吞吐和显存；
- [ ] 检查评测与压测是否可重复。

产物：第一版 `inference-benchmark.md`。

### 第 4 周：量化、分布式与服务闭环

**第 22 天：量化对比**

- [ ] 比较 BF16、INT8、INT4；
- [ ] 运行相同质量回归集；
- [ ] 记录加载时间、显存和吞吐。

产物：量化对照表。

**第 23 天：调度与混合流量**

- [ ] 设计短输入、长输入、长输出三类请求；
- [ ] 测试不同并发；
- [ ] 观察长短请求混跑的尾延迟。

产物：调度实验记录。

**第 24 天：并行策略设计**

- [ ] 画出 DDP、FSDP/ZeRO、TP 的状态切分；
- [ ] 预测每种配置的显存和通信；
- [ ] 固定多卡实验矩阵。

产物：实验假设和预期表。

**第 25 天：多 GPU 实验**

- [ ] 短租 2～4 卡；
- [ ] 测量集合通信；
- [ ] 使用同一个约 1B 模型对比全参数 DDP 与 FSDP/ZeRO；
- [ ] 使用 7B/8B 主模型运行一次 Tensor Parallel 推理。

产物：原始 trace 和指标。

**第 26 天：多卡分析与服务治理**

- [ ] 分析 step time 和通信等待；
- [ ] 增加并发限制、队列、超时和取消；
- [ ] 暴露 Prometheus 指标。

产物：`distributed-comparison.md` 和服务治理配置。

**第 27 天：故障实验**

- [ ] 注入超长输入、OOM、Worker 退出和客户端中断；
- [ ] 检查错误响应、日志和指标；
- [ ] 验证服务恢复和请求清理。

产物：故障实验记录。

**第 28 天：最终复现**

- [ ] 在干净环境重建依赖；
- [ ] 复现一个训练 Checkpoint；
- [ ] 执行质量回归和推理压测；
- [ ] 完成最终报告和后续学习清单。

产物：`final-report.md`。

## 9. 必做实验矩阵

### 9.1 训练矩阵

| 实验 | 对照变量 | 固定变量 | 核心指标 |
|---|---|---|---|
| 精度 | FP32 / BF16 | 模型、数据、batch tokens | step time、显存、Loss |
| 激活重算 | 开 / 关 | 模型、精度、batch tokens | 显存、step time |
| Batch 策略 | micro-batch / accumulation | 有效 batch tokens | 吞吐、显存、收敛 |
| LoRA Rank | 至少两个 rank | 数据、步数、target modules | 质量、参数量、显存 |

### 9.2 推理矩阵

| 维度 | 至少覆盖 |
|---|---|
| 引擎 | Transformers、vLLM |
| 精度 | BF16、INT8、INT4 |
| 输入长度 | 512、2048、8192 |
| 并发 | 1、8、32、64 |
| 流量 | 短请求、长请求、长短混合 |
| 指标 | TTFT、TPOT、P50/P95/P99、tokens/s、显存、错误率 |

### 9.3 分布式矩阵

| 实验 | 目的 |
|---|---|
| 单卡基线 | 固定吞吐和显存基准 |
| 约 1B 模型、2 卡 DDP | 测量数据并行扩展效率 |
| 同一约 1B 模型、2 卡 FSDP/ZeRO | 测量状态分片的显存收益和通信代价 |
| 7B/8B 模型、2 卡 TP 推理 | 验证单模型跨卡切分和延迟变化 |
| Worker 失败 | 观察任务、进程组和服务的实际失败边界 |

## 10. 最终代码实践清单

### 模型与训练

- [ ] 不依赖 Transformers 实现 Decoder-only Transformer；
- [ ] 测试 Causal Mask、RoPE、GQA 和 LM Loss；
- [ ] 完成小模型预训练；
- [ ] 支持 Checkpoint 保存与恢复；
- [ ] 完成 7B/8B QLoRA；
- [ ] 完成一次小规模 DPO；
- [ ] 记录显存、吞吐、梯度和数值异常。

### 数据与质量

- [ ] 固定训练、验证和回归集；
- [ ] 检查重复、泄漏、长度和格式；
- [ ] 保存原始模型输出；
- [ ] 同时报告改善与退化；
- [ ] 任何自动评分都可以追溯到样本；
- [ ] 不使用训练 Loss 替代任务质量。

### 推理与服务

- [ ] 实现带 KV Cache 的自回归生成；
- [ ] 部署 Transformers 基线；
- [ ] 部署 vLLM；
- [ ] 完成多并发、长上下文和混合流量压测；
- [ ] 完成 BF16、INT8、INT4 对比；
- [ ] 暴露服务指标；
- [ ] 实现限流、超时、取消和过载保护。

### 分布式

- [ ] 测量集合通信；
- [ ] 对比 DDP 与 FSDP/ZeRO；
- [ ] 完成一次 Tensor Parallel 推理；
- [ ] 计算并验证扩展效率；
- [ ] 保存 profiler trace；
- [ ] 执行 Worker 故障实验。

### 可复现性

- [ ] 训练和服务依赖分别锁定；
- [ ] 配置、代码 commit、数据版本和模型版本可追踪；
- [ ] 大文件有路径和校验和；
- [ ] 干净环境可重建；
- [ ] 一条命令执行质量回归；
- [ ] 一条命令执行推理压测。

## 11. 最终验收

完成不以“12 天结束”为准，而以以下结果为准：

1. 最小模型能训练、恢复并生成可观察改善的文本。
2. 7B/8B QLoRA 在 32GB GPU 内稳定运行。
3. 微调收益通过固定评测集证明，并包含退化分析。
4. KV Cache 估算与实际显存趋势一致。
5. vLLM 相对 Transformers 的收益有并发和请求长度条件说明。
6. 量化结论同时包含质量、延迟、吞吐和显存。
7. 至少完成一次真实多 GPU 对照实验。
8. 服务过载和 Worker 失败的表现有原始日志和指标。
9. 新环境能够按照 README 复现关键实验。
10. 最终报告明确列出仍不理解、未验证和需要进一步深入的部分。

## 12. 后续 30 天方向

完成本计划后，只选择一个方向继续深入：

- **训练系统**：FSDP2、DeepSpeed、Sequence Parallel、Context Parallel、Checkpoint IO；
- **推理系统**：vLLM 调度器、Prefix Cache、Speculative Decoding、分布式 KV Cache；
- **Kernel**：Triton、FlashAttention、融合算子、量化 Kernel；
- **评测与数据**：数据治理、污染检测、Judge 校准、在线反馈闭环；
- **生产架构**：多模型路由、弹性伸缩、SLO、成本模型和容量规划。

优先根据 12 天实验中暴露的真实短板选择，而不是同时扩展所有方向。
