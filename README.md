# Laya Multilingual Playground

WSL 下运行 Laya Multilingual 的最小实验项目：ModelScope 下载 → 本地目录 → Laya SDK → 中文输入 → 结构化决策 + 概率。

```text
Model:      convaiinnovations/laya-multilingual
Source:     ModelScope
Model type: Non-autoregressive System 1 Decision Model
Language:   100+ languages
Backend:    PyTorch + Transformers
Runtime:    CPU (WSL2)
Ollama:     Not required
```

没有 FastAPI、没有 Docker、没有 Web UI。整个仓库只做一件事：把
`State + Typed Question → Decision + Probability` 这条链路跑通。

---

## Quickstart

```bash
git clone <this repo>
cd laya-multilingual-playground

make setup                       # uv venv --python 3.11 && uv sync --extra dev
source .venv/bin/activate

export USE_TF=0                  # 见 "TensorFlow 死锁" 一节

make download                    # ModelScope -> models/laya-multilingual (~647 MiB)
make demo                        # 中文工单 -> choice
```

其它入口：

```bash
make typed                       # choice + score + noul，一次 forward pass
make compare                     # 10 条中文场景横向对比
make benchmark                   # 本地 cold start / warm inference 延迟
make test                        # pytest smoke tests
```

不用 make 也可以，命令是普通的 `uv run`：

```bash
uv run python scripts/download_model.py
uv run python examples/basic_multilingual.py
```

---

## Laya 不是 LLM

这是本项目最重要的区别。

```text
LLM:

Input
 ↓
Generation
 ↓
Text
 ↓
Parse
 ↓
Action


Laya:

State
 +
Typed Questions
 ↓
Decision Model
 ↓
Structured Decision
 +
Probability
```

`Laya does not generate a long answer.` 它没有解码循环，输出直接从「选项标记位置」的
分数做 softmax，所以：

- 没有 parse 环节，也就没有 parse 失败；
- 不会生成多余文本，没有可幻觉的内容；
- 一次 forward pass 同时回答多个 typed question；
- 每个答案自带概率和 confidence。

三种 primitive：

| Primitive | 返回字段 | 用途 |
|---|---|---|
| `choice` | `choice` / `probabilities` / `confidence` | 部门路由、意图分类 |
| `score` | `score` / `legend` / `probabilities` / `confidence` | 紧急度、严重度（有序等级） |
| `noul` | `noul`（P(true））/ `confidence` | 是否成立：退款意图、风险、是否相关 |

调用形状（`examples/basic_multilingual.py`）：

```python
state = {"subject": "订单重复扣款",
         "body": "我的订单昨天被扣了两次钱，希望尽快退款。"}

questions = {"department": {
    "type": "choice",
    "instructions": "这个工单应该交给哪个部门处理？",
    "criteria": {"billing": "账单、扣款、退款、发票", "technical": "报错、崩溃、故障、无法运行",
                 "logistics": "发货、快递、配送、物流延迟", "account": "账号、登录、密码、权限",
                 "other": "咨询、介绍、不属于以上任何一类"}}}

result = agent.predict(state, questions)
result["answers"]["department"]["choice"]          # -> "billing"
result["answers"]["department"]["probabilities"]   # -> {"billing": 0.9964, ...}
```

一条 state + 三个 typed questions = 一次 forward pass，见 `examples/typed_decisions.py`。

---

## Does this require Ollama?

No.

Laya is a decision model, not a text-generation model.

This project does not require:

- Ollama
- Qwen
- Llama
- OpenAI API
- Anthropic API

Ollama can be integrated later as a reasoning/generation component, but it is not required for this experiment.

本项目不安装、不调用 Ollama，也不下载任何生成式模型。后续怎么接，见
[docs/architecture.md](docs/architecture.md)。

---

## Model Source

Primary:
ModelScope

Fallback / Reference:
Hugging Face

```text
ModelScope:   https://www.modelscope.cn/models/convaiinnovations/laya-multilingual
Hugging Face: https://huggingface.co/convaiinnovations/laya-multilingual
```

本项目为了中国网络环境与模型管理便利，优先从 ModelScope 下载模型。

本次实验做过逐文件 SHA-256 比对：ModelScope 与 Hugging Face 两边**同名文件字节级一致**，
包括 614 MiB 的权重（本地 `sha256sum` = ModelScope API 的 `Sha256` = HF 的 LFS `oid` =
`9d628fd9…5aa8f204`），`tokenizer/tokenizer.json` 同样一致。差异只有一处：ModelScope 多了一个
ModelScope 自己的元数据文件 `configuration.json`。

这只是本次快照（ModelScope revision `5ad6e84d…`）的比对结果，**不代表两个仓库会持续同步**；
重新下载后请再跑一次完整性检查。

### 实际加载路径：ModelScope → 本地目录 → Laya

`laya.load()` 的第一个参数就叫 `model_id_or_path`，实现里先判断路径是否存在，
只有不存在时才去 hub 下载（`laya/agent.py`，`Agent.__init__`）：

```python
model_dir = model_id_or_path
if not os.path.exists(model_dir):      # 本地存在就完全不碰 hub
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(model_id_or_path, ...)
```

所以本项目做的是：

```python
agent = laya.load("models/laya-multilingual")   # 绝对路径，来自 ModelScope
```

不需要任何兼容层，也没有改 Laya 包内部源码。

这条路径经过验证：把 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 打开后加载和推理照常工作，
且机器上不存在 `~/.cache/huggingface` 目录 —— 权重、`encoder/config.json`、
`tokenizer/` 全部来自 ModelScope 下载的目录，加载过程不访问 Hugging Face。

### 模型完整性检查

`scripts/download_model.py` 下载后逐个文件检查（不是只检查目录存在），缺文件就 `exit 1`：

```text
README.md · model.safetensors · rl_agent_config.json
encoder/config.json · tokenizer/tokenizer.json · tokenizer/tokenizer_config.json
```

---

## TensorFlow 死锁

Laya 官方 README 说明：如果环境里装了 TensorFlow，`transformers` 在 import 时探测 TF，
其 abseil runtime 可能让模型构建 deadlock。

对策是在 **import laya 之前** 设置 `USE_TF=0`：

```python
import os
os.environ.setdefault("USE_TF", "0")   # 必须在 import laya 之前
import laya
```

本项目里这件事已经做掉：`src/laya_play/__init__.py` 第一行就设置该变量，`Makefile` 里
`export USE_TF := 0` 覆盖所有入口。手工跑 python 时记得 `export USE_TF=0`。

本环境没有安装 TensorFlow，所以这里只是预防性措施。

---

## Results（本机实测，WSL2 + CPU）

### 中文工单 → choice

```text
Input:      我的订单昨天被扣了两次钱，希望尽快退款。
Decision:   department = billing
Confidence: 0.9822
Probs:      billing 0.9964 | technical 0.0007 | logistics 0.0016 | account 0.0006 | other 0.0006
Latency:    ~129 ms (CPU, warm)
```

### 10 条中文场景（`make compare`）

department = `choice`；urgency = `score`（0-based 期望值）；ai? = `noul`，P(「与 AI 模型部署有关」)。

| 输入 | decision | conf | urgency | ai? | ms |
|---|---|---|---|---|---|
| 我的订单昨天被扣了两次钱，希望尽快退款。 | billing | 0.975 | 1.87 | 0.97 | 350 |
| 连不上公司网络，客户端一直提示连接超时… | technical | 0.945 | 1.65 | 0.91 | 369 |
| 我们部署的大模型服务启动失败，日志显示权重加载异常 | technical | 0.855 | 2.79 | 0.92 | 332 |
| 官网页面一直转圈打不开，换浏览器刷新多次也没用 | technical | 0.965 | 3.24 | 0.97 | 315 |
| 订单显示已发货，但三天没有物流更新 | logistics | 0.969 | 1.18 | 0.35 | 326 |
| 我修改密码之后无法登录，提示账号或密码错误 | account | 0.989 | 1.92 | 0.68 | 335 |
| 商品还没有发货，我想取消订单并申请全额退款 | logistics | 0.615 | 1.49 | 0.29 | 321 |
| 模型推理接口每次要等十几秒，batch size 调小也没改善 | technical | 0.855 | 2.09 | 0.77 | 328 |
| 训练时报 CUDA out of memory，24G 显存不够用 | technical | 0.496 | 2.63 | 0.88 | 338 |
| 你们的产品支持哪些语言？有没有企业版？ | other | 0.956 | 1.43 | 0.72 | 317 |

读这张表要注意两件事，都是这个 checkpoint 的真实属性，不是 bug：

1. `choice` 在 10 条里明显合理 8 条。「退款」被判成 `logistics`（0.756 vs billing 0.232），
   因为句子里「还没发货」压过了「退款」。这是语义竞争，不是随机噪声。
2. `noul` 完全不可信：和 AI 部署无关的「页面打不开」也给 0.97。官方模型卡写明
   `laya-multilingual` 在 typed-decisions 上 zero-shot 0.342 ≈ 随机（0.318），
   且**未做温度校准**，systematically over-confident。别把这里的概率当概率用。

结论：这个 checkpoint 现在能用的部分是 `choice`；`score` / `noul` 要在自己的数据上做
fine-tune / 温度校准之后才有意义。

### Local WSL benchmark（`make benchmark`）

```text
cold start (laya.load of the local ModelScope checkpoint)  ~20.5 s
warm inference, 1 question:   min 123.6 | p50 128.7 | avg 129.2 | max 144.7 ms
questions per request:        1 → 123.8 ms · 3 → 340.9 ms · 10 → 986.2 ms (98.6 ms/question)
```

环境：WSL2 (Linux 6.6.114.1-microsoft-standard-WSL2)，x86_64，16 logical CPUs，
`torch.get_num_threads() = 8`，torch 2.14.0+cpu，float32（CPU 上 Laya 强制 fp32）。

### Official benchmark

Official Laya numbers are measured on a **Tesla T4 GPU** and are **not** local numbers:

```text
laya-multilingual, 1 question   32.8 ms
laya-multilingual, 10 questions 72.3 ms (7.2 ms/question)
```

本项目的 ~129 ms 是 CPU 数字，两者不可比较。要拿 GPU 数字得换 CUDA 版 torch
（见 docs/troubleshooting.md）。

---

## Layout

```text
laya-multilingual-playground/
├── README.md
├── IMPLEMENTATION_REPORT.md      实测记录：版本、体积、延迟、踩坑
├── pyproject.toml / uv.lock
├── Makefile
├── conftest.py                   给 pytest 兜底设置 USE_TF=0
├── models/laya-multilingual/     权重（gitignored）
├── scripts/download_model.py     ModelScope 下载 + 逐文件完整性检查
├── src/laya_play/                共用 glue：路径、完整性、load、计时、打印
│   └── questions.py              中文工单 state + typed question schema
├── examples/
│   ├── basic_multilingual.py     choice
│   ├── typed_decisions.py        choice + score + noul，一次 forward pass
│   ├── compare_inputs.py         10 条中文场景
│   └── benchmark.py              cold start / warm inference
├── tests/test_smoke.py           13 个结构断言，不断言具体概率
└── docs/
    ├── architecture.md           后续 Ollama + Laya + Tools 的位置
    └── troubleshooting.md        真实踩过的坑与诊断顺序
```

模型权重不进 Git：`.gitignore` 里 `models/*` + `!models/.gitkeep`。

---

## Versions（实测）

```text
uv            0.11.6
Python        3.11.15        (uv-managed CPython, 不依赖系统 3.10)
laya          0.3.4
torch         2.14.0+cpu     (PyTorch CPU index, 见 pyproject.toml)
transformers  5.17.0
safetensors   0.8.0
modelscope    1.40.1  (modelscope-hub 0.4.5)
numpy         2.4.6
pytest        9.1.1
```

`torch` 走的是 `https://download.pytorch.org/whl/cpu` 显式索引，避免在纯 CPU 实验里拖
2 GB+ 的 CUDA wheel。要用 GPU 见 docs/troubleshooting.md。

---

## Next

第一阶段只到 Laya 为止。后续接 Ollama / Agent / MCP / Model Router 的架构图和分工说明在
[docs/architecture.md](docs/architecture.md)；本阶段不实现它们。

Apache 2.0（与 Laya 一致）· Convai Innovations 提供模型，本项目只是使用示例。
