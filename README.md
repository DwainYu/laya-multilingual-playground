# Laya Multilingual Playground

A hands-on playground for running **Laya Multilingual**, a non-autoregressive *System 1 decision
model*, locally on WSL2 CPU — model pulled from ModelScope, loaded from a project-local
directory, and asked typed questions in Chinese.

```text
Model:      convaiinnovations/laya-multilingual
Source:     ModelScope
Model type: Non-autoregressive System 1 Decision Model
Language:   100+ languages
Backend:    PyTorch + Transformers
Runtime:    WSL2 + CPU
Ollama:     Not required
Tests:      13 passed
```

这是「Laya Multilingual 决策模型」的最小可复现实验项目，不是聊天机器人，也不做完整 Agent。
没有 FastAPI、没有 Docker、没有 Web UI。整个仓库只把一条链路跑通：

```text
State + Typed Question  →  Decision + Probability
```

---

## Overview

Laya is a **decision model**, not a text-generation model. You give it a *state* (text, ticket,
email or JSON) plus *typed questions* (`choice` / `score` / `noul`), and it returns structured
answers with probabilities in **a single forward pass**. There is no decoding loop, so there is
nothing to parse and nothing to hallucinate.

Verified end to end on this machine:

```text
WSL2  →  uv (Python 3.11.15)  →  ModelScope snapshot_download
      →  models/laya-multilingual/  →  laya.load(<local path>)
      →  Chinese input  →  typed decision  →  choice / score / noul  +  probability
```

---

## Experiment Results

Every number below was measured on this WSL2 + CPU machine; see
[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for the full log. Latency figures are ranges
across the repeat runs actually performed — on WSL2 CPU they drift roughly ±10 % between runs,
so treat them as an order of magnitude, not a stable baseline.

| Item | Result |
|---|---|
| Environment | WSL2 + CPU |
| Python | 3.11.15 |
| laya | 0.3.4 |
| transformers | 5.17.0 |
| torch | 2.14.0+cpu |
| modelscope | 1.40.1 |
| Model | laya-multilingual |
| Model size | 646.8 MiB |
| Download | ~25 s @ ~33 MB/s |
| Demo latency (first predict in a fresh process) | 139–189 ms |
| Warm avg, 1 question/request (2 runs of 10) | 129.2 / 140.5 ms |
| Cold start (`laya.load`) | 20.5–22.6 s |
| Tests | 13 passed |

Official Laya figures (32.8 ms / 1 question) are measured on a **Tesla T4 GPU** and are **not**
comparable to the CPU numbers above. Local and official benchmarks are kept in separate sections
on purpose.

---

## Known Limitation / Experiment Finding

**Important finding:**

Choice-based decisions were reasonably useful in the Chinese test set, but the current
multilingual checkpoint's zero-shot **Noul** behavior was unreliable.

For example, the question *"Is this request related to AI model deployment?"* (`noul`) returned a
high probability even for clearly unrelated requests — a pure refund ticket got `noul = 0.9409`,
and "the web page won't open" got `0.97`.

This is consistent with the current checkpoint documentation: the zero-shot typed-decision score
(0.342) is close to the random baseline (0.318), and the temperature parameters ship
uncalibrated (`temperature = [1.0, 1.0, 1.0]`), so the model is systematically over-confident.

Wording matters here: this is **not** "Laya Noul is broken". It is
*the current checkpoint showed unreliable zero-shot Noul behavior in this local experiment.*
Practical consequence for this repo:

- `choice` — usable today (8 of 10 Chinese cases clearly sensible).
- `score` — the weakest primitive; read it as a distribution, not a verdict.
- `noul` — do **not** gate on it until you fine-tune on your own data and fit temperatures.

Do not trust these probabilities before calibrating on your own labelled data.

---

## Installation

```bash
git clone https://github.com/DwainYu/laya-multilingual-playground.git
cd laya-multilingual-playground

uv sync
source .venv/bin/activate

export USE_TF=0        # must be set BEFORE importing laya — see "TensorFlow deadlock"
```

Then download the checkpoint and run the experiment:

```bash
make download          # ModelScope -> models/laya-multilingual (646.8 MiB, ~25 s)
make demo              # Chinese ticket -> choice decision
make test              # 13 structural smoke tests
```

Other entry points:

```bash
make typed                       # choice + score + noul in ONE forward pass
make compare                     # 10 Chinese scenarios side by side
make benchmark                   # local cold-start + warm-inference latency
make help                        # list every target
```

Without `make` (same commands the Makefile wraps):

```bash
uv run python scripts/download_model.py
uv run python examples/basic_multilingual.py
uv run pytest -v
```

`make setup` does the `uv venv --python 3.11` + `uv sync --extra dev` pair in one step if you
prefer not to type it. Model weights are never committed (`.gitignore`: `models/*`,
`!models/.gitkeep`).

---

## Does this require Ollama?

No. **Ollama is NOT required.**

Laya is a decision model, not a text-generation model. This project does not use Ollama or a
text-generation LLM, and does not require:

- Ollama
- Qwen
- Llama
- OpenAI API
- Anthropic API

Ollama can be integrated later as a reasoning/generation component, but it is not required for
this experiment. 本项目全程不安装、不调用 Ollama。

---

## Model

```text
convaiinnovations/laya-multilingual
```

- ModelScope (primary): <https://www.modelscope.cn/models/convaiinnovations/laya-multilingual>
- Hugging Face (reference): <https://huggingface.co/convaiinnovations/laya-multilingual>

本项目为了中国网络环境与模型管理便利，优先从 ModelScope 下载模型。

The ModelScope and Hugging Face files were verified file-by-file using SHA-256 in this
experiment, including the 614 MiB weights
(`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204` — local `sha256sum` =
ModelScope API `Sha256` = HF LFS `oid`). That is a statement about **this snapshot**
(ModelScope revision `5ad6e84d…`), not a claim that the two repositories stay byte-synchronised;
re-run the integrity check after any re-download.

### Actual load path: ModelScope → local directory → Laya

`laya.load()`'s first parameter is literally `model_id_or_path`, and the implementation checks
for a local path first — it only reaches for a hub download when the path does not exist
(`laya/agent.py`, `Agent.__init__`):

```python
model_dir = model_id_or_path
if not os.path.exists(model_dir):      # local directory present → hub is never touched
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(model_id_or_path, ...)
```

So this project simply does:

```python
agent = laya.load("models/laya-multilingual")   # absolute path, downloaded from ModelScope
```

No compatibility layer, and no patching of Laya's own source. Verified: with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` set, load and inference still work, and
`~/.cache/huggingface` does not exist on this machine — weights, `encoder/config.json` and
`tokenizer/` all come from the ModelScope directory.

### Integrity check

`scripts/download_model.py` checks required files **individually** (not just that the directory
exists) and exits 1 with the missing list:

```text
README.md · model.safetensors · rl_agent_config.json
encoder/config.json · tokenizer/tokenizer.json · tokenizer/tokenizer_config.json
```

---

## Architecture

```text
User / State
     ↓
Laya Multilingual          ← laya.load("models/laya-multilingual")
     ↓
Typed Questions            ← choice / score / noul
     ↓
Choice / Score / Noul
     ↓
Decision + Probability     ← result["answers"][name]
```

One shared module (`src/laya_play/`) owns paths, integrity checks, loading, timing and printing;
`examples/` and `tests/` depend on it and never re-implement loading.
Details in [docs/architecture.md](docs/architecture.md).

### Future work — not implemented here

```text
Ollama  +  Laya  +  Tools / MCP
```

The intended split: Ollama does generative reasoning and planning, Laya does fast structured
decisions and gating, and Code executes the actual side effects. The first concrete step is a
confidence gate on Laya's output. None of it exists in this repo yet, and the current checkpoint
is not calibrated enough for a gate to be trustworthy — that ordering is documented in
[docs/architecture.md](docs/architecture.md).

---

## Project Structure

```text
laya-multilingual-playground/
├── README.md
├── IMPLEMENTATION_REPORT.md      full measured log: versions, sizes, latency, gotchas
├── PUBLISH_REPORT.md             how this repo was published + clean-clone repro check
├── LICENSE
├── docs/
│   ├── architecture.md           current pipeline + where Ollama/Tools would fit
│   └── troubleshooting.md        real failures hit here, plus a diagnostic order
├── examples/                     basic_multilingual · typed_decisions · compare_inputs · benchmark
├── src/laya_play/                shared glue + Chinese question schemas
├── tests/                        test_smoke.py (13 structural tests)
├── scripts/download_model.py     ModelScope download + per-file integrity check
├── models/                       checkpoint lands here (gitignored)
├── Makefile
├── pyproject.toml
└── uv.lock
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

`Laya does not generate a long answer.` 它没有解码循环，输出直接从「选项标记位置」的分数做
softmax，所以：

- 没有 parse 环节，也就没有 parse 失败；
- 不会生成多余文本，没有可幻觉的内容；
- 一次 forward pass 同时回答多个 typed question；
- 每个答案自带概率和 confidence。

三种 primitive：

| Primitive | 返回字段 | 用途 |
|---|---|---|
| `choice` | `choice` / `probabilities` / `confidence` | 部门路由、意图分类 |
| `score` | `score` / `legend` / `probabilities` / `confidence` | 紧急度、严重度（有序等级） |
| `noul` | `noul`（P(true)）/ `confidence` | 是否成立：退款意图、风险、是否相关 |

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

## TensorFlow 死锁

Laya 官方 README 说明：如果环境里装了 TensorFlow，`transformers` 在 import 时探测 TF，其
abseil runtime 可能让模型构建 deadlock。

对策是在 **import laya 之前** 设置 `USE_TF=0`：

```python
import os
os.environ.setdefault("USE_TF", "0")   # 顺序错了就没用
import laya
```

本项目里这件事已经做掉：`src/laya_play/__init__.py` 第一行就设置该变量，`Makefile` 里
`export USE_TF := 0` 覆盖所有入口。手工跑 python 时记得 `export USE_TF=0`。本环境没有安装
TensorFlow，所以这里只是预防性措施。

---

## Results（本机实测，WSL2 + CPU）

### 中文工单 → choice

```text
Input:      我的订单昨天被扣了两次钱，希望尽快退款。
Decision:   department = billing
Confidence: 0.9822
Probs:      billing 0.9964 | technical 0.0007 | logistics 0.0016 | account 0.0006 | other 0.0006
Latency:    139 ms（进程内第一次 predict，之后每次 ~129 ms）
```

### 10 条中文场景（`make compare`）

department = `choice`；urgency = `score`（0-based 期望值）；ai? = `noul`，P(「与 AI 模型部署有关」)。

| 输入 | decision | conf | urgency | ai?（noul） | ms |
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

读这张表要注意两件事，都是这个 checkpoint 的真实属性，不是 bug（详见上面
**Known Limitation / Experiment Finding**）：

1. `choice` 在 10 条里明显合理 8 条。「退款」被判成 `logistics`（0.756 vs billing 0.232），因为
   句子里「还没发货」压过了「退款」。这是语义竞争，不是随机噪声。
2. `noul` 这一列整体不可信：和 AI 部署无关的「页面打不开」也给 0.97。官方模型卡写明
   `laya-multilingual` 在 typed-decisions 上 zero-shot 0.342 ≈ 随机（0.318），且**未做温度校准**，
   systematically over-confident。别把这里的数字当概率用。

### Local WSL benchmark（`make benchmark`）

Two runs of `make benchmark` were performed (first the experiment, then again right before
this release):

```text
                                          run A (experiment)   run B (pre-release)
cold start  (laya.load of local ckpt)     20.53 s              22.57 s
warm 1 question   min                     123.62 ms            132.53 ms
                  p50                     128.71 ms            140.45 ms
                  avg                     129.24 ms            140.50 ms
                  max                     144.66 ms            148.18 ms
1 / 3 / 10 questions per request          123.8 / 340.9 /      149.3 / 315.6 /
                                          986.2 ms             996.0 ms
```

环境：WSL2 (Linux 6.6.114.1-microsoft-standard-WSL2)，x86_64，16 logical CPUs，
`torch.get_num_threads() = 8`，torch 2.14.0+cpu，float32（CPU 上 Laya 强制 fp32）。
两次运行相差约 ±8–10%，别把任何一个数字当稳定基线。

### Official benchmark

Official Laya numbers are measured on a **Tesla T4 GPU** and are **not** local numbers:

```text
laya-multilingual, 1 question   32.8 ms
laya-multilingual, 10 questions 72.3 ms (7.2 ms/question)
```

本项目的 ~129 ms 是 CPU 数字，两者不可比较。要拿 GPU 数字得换 CUDA 版 torch
（见 [docs/troubleshooting.md](docs/troubleshooting.md)）。

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

`torch` 走的是 `https://download.pytorch.org/whl/cpu` 显式索引，避免在纯 CPU 实验里拖 2 GB+ 的
CUDA wheel。本机的 `nvidia-smi` 能看到 RTX 3060，但 `+cpu` 版 torch 用不到它——这是刻意的：
第一阶段目标就是 WSL2 + CPU 跑完全部实验。

---

## License

This repository's code is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).

```text
Copyright 2026 DwainYu

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions
and limitations under the License.
```

The `convaiinnovations/laya-multilingual` checkpoint is also Apache 2.0, provided by
Convai Innovations; this repository only downloads and uses it, and ships no weights.
