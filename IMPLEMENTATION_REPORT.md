# Implementation Report

Laya Multilingual Playground · 实际执行记录。所有数字都是这台 WSL2 机器上真实跑出来的，
不是引用官方数据。日期：2026-09-21。

---

## 1. 实际版本

```text
OS               Linux 6.6.114.1-microsoft-standard-WSL2  x86_64   (WSL2)
CPU              16 logical
Python           3.11.15        uv 管理的 CPython，系统 python3 是 3.10.12，未被使用
uv               0.11.6
laya             0.3.4
modelscope       1.40.1         (+ modelscope-hub 0.4.5)
torch            2.14.0+cpu
transformers     5.17.0
safetensors      0.8.0
numpy            2.4.6
pytest           9.1.1
虚拟环境         .venv (uv venv --python 3.11)，uv.lock 已提交
```

`laya.load` 的真实签名（introspection 结果，不是猜的）：

```text
(model_id_or_path: str = 'convaiinnovations/laya', device: Optional[str] = None,
 token: Optional[str] = None, subfolder: Optional[str] = None) -> laya.agent.Agent
```

## 2. 模型

```text
Model ID    convaiinnovations/laya-multilingual
来源         ModelScope  https://www.modelscope.cn/models/convaiinnovations/laya-multilingual
revision    5ad6e84d70d70edde0d5a3241c5233d0b11cd6b8
本地路径     models/laya-multilingual/          （绝对路径
                                              /home/user/projects/laya-multilingual-playground/models/laya-multilingual）
总体积       646.8 MiB（du -sh: 647M）
下载耗时     25 s，峰值 ~33 MB/s
```

| 文件 | 大小 |
|---|---|
| `model.safetensors` | 614.01 MiB（643,835,514 B） |
| `tokenizer/tokenizer.json` | 32.77 MiB |
| `README.md` | 6.9 KB |
| `encoder/config.json` | 1.9 KB |
| `rl_agent_config.json` | 472 B |
| `tokenizer/tokenizer_config.json` | 502 B |
| `configuration.json`（ModelScope 独有） | 77 B |

参数 321,908,995（F16 321,908,995 + F32 3），encoder `jhu-clsp/mmBERT-base`
（ModernBERT 架构：22 层 / hidden 768 / vocab 256,000），`max_len 1024`，`head_max_len 256`，
`temperature [1.0, 1.0, 1.0]`（出厂未校准）。

**ModelScope / Hugging Face 一致性**：逐文件 SHA-256 比对，同名文件字节级相同。
本地 `sha256sum(models/laya-multilingual/model.safetensors)` =
`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204` =
ModelScope API 的 `Sha256` = HF LFS `oid`。唯一差异是 ModelScope 多出 `configuration.json`。
这只是本次快照的结论，不代表两仓持续同步。

## 3. 设备

```text
Runtime device: CPU
CUDA available: no        torch.cuda_is_available() is False
```

这是刻意选择：`pyproject.toml` 用显式索引把 `torch` 锁到
`https://download.pytorch.org/whl/cpu`，避免在纯 CPU 实验里拖 2 GB+ 的 CUDA wheel。
CPU 上 Laya 会强制 float32（源码 `Agent.__init__`：`device.type in ("cpu","mps") → dtype = float32`），
所以内存占用 ~1.3 GB，7 GiB 的 WSL 实例没问题。

本机 `nvidia-smi` 其实能看到 RTX 3060 Laptop 6 GiB，但 `+cpu` 版 torch 用不到它。
要 GPU 就换 cu128 wheel（`docs/troubleshooting.md`），届时延迟数字必须重测。

## 4. Demo 输出摘要

### `make demo` — 中文工单 → choice

```text
Input:      我的订单昨天被扣了两次钱，希望尽快退款。
Decision:   department = billing
Confidence: 0.9822
Probs:      billing 0.9964 / technical 0.0007 / logistics 0.0016 / account 0.0006 / other 0.0006
Latency:    139–189 ms（进程内第一次 predict，之后每次 ~129 ms）
Status:     PASS
usage:      {"input_tokens": 106, "output_tokens": 0}
```

### `make typed` — 一条 state + 三个 typed questions，一次 forward pass

| question | type | 结果 |
|---|---|---|
| `department` | choice | `billing`，conf 0.9822 |
| `urgency` | score | 期望值 1.8533（0-based），argmax = level 3/5「紧急」p=0.5487，conf 0.3193 |
| `ai_deployment_related` | noul | 0.9409 |

```text
Latency: 320.10 ms for 3 questions (106.70 ms/question)
usage:   {"input_tokens": 256, "output_tokens": 0}
```

三个问题共享一次 forward pass，`output_tokens` 恒为 0 —— 没有任何文本生成，这就是
System 1 决策模型和 LLM 的区别所在。

### `make compare` — 10 条中文场景

完整表在 README。**`choice` 10 条里 8 条明显合理**；两个判错的很有信息量：

- 「商品还没发货，我想取消订单并申请全额退款」→ `logistics` 0.756（应为 billing），
  「未发货」的语义压过了「退款」。
- 「GPU 显存不足」conf 只有 0.496 —— 模型自己也犹豫，这类正是该走 confidence gate 的。

`noul`（是否与 AI 模型部署有关）**完全不可用**：「页面打不开」也给 0.97。与官方声明一致
（zero-shot typed-decisions 0.342 ≈ 随机 0.318，且未校准）。

## 5. Benchmark（本地，CPU）

```text
cold start (laya.load，读本地 ModelScope 目录)   20.53 s
warm inference, 1 question/request (10 次)      min 123.62 | p50 128.71 | avg 129.24 | max 144.66 ms
  prediction 1/2/3 = 144.66 / 133.88 / 129.68 ms（第一次最慢，之后稳定在 ~126 ms）
questions per request                           1 → 123.78 ms | 3 → 340.91 ms | 10 → 986.22 ms (98.62 ms/q)
typed 3-question 单次 forward pass               308.72 ms
compare 10 条输入（各 3 questions）              min 315.3 | avg 333.2 | max 369.0 ms
环境                                            16 logical CPUs，torch threads=8，float32
```

run-to-run 波动实测约 ±10%（第二次跑同样的 1-question 是 141.21 ms，3-question 356.91 ms，
10-question 1084.33 ms）。WSL 上 CPU 频率与宿主负载会直接影响这些数字，别把它们当稳定基线。

官方 laya-multilingual 是 T4 GPU 上 32.8 ms（1 question）/ 72.3 ms（10 questions）。
**本地 CPU 数字约为其 4 倍（单问）到 14 倍（10 问）**，两者不可互换引用。README 把
Official benchmark 和 Local WSL benchmark 分成两节。

批量收益很小（123.8 → 98.6 ms/question）是因为每条 question 都要占满 `max_len=1024`
的序列，padding 主导了成本；官方在 GPU 上的放大比例明显更高。

## 6. 测试

```bash
$ make test
13 passed in 22.41s
```

覆盖：`import laya` → 模型目录存在 → **6 个必需文件逐个存在**（parametrize）→
`missing_files() == []` → 设备字段 → `predict` 返回 dict → `answers` 键集合等于请求的
question 名 → choice/score/noul 各自字段与取值范围 → 概率和 ≈ 1 → 中文输入得到合法标签。

没有任何一条断言写死具体概率（模型未校准，写了就是脆弱测试）。

## 7. 遇到的问题与解决

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `uv sync` 失败：`OSError: Readme file does not exist: README.md` | hatchling 构建 editable 包时校验 `readme` 字段，而当时 README.md 还没写 | 先写 README.md；已在文档里记录 |
| 2 | 环境没有 `unzip`，`find -exec` 被 rtk 包装层拒绝 | 容器里的工具裁剪 + find 被替换 | 改用 `python -m zipfile` 展开 wheel 读源码 |
| 3 | `TypeError: unsupported format string passed to PosixPath.__format__` | f-string 对 `Path` 用了 `:<32` 对齐 | `str(path.relative_to(target))` |
| 4 | 第二次 `make download` 重复拉 `tokenizer_config.json` | `laya.load()` 会就地改写该文件（`_fix_tokenizer_config`：`extra_special_tokens` 由 list 改成 dict，transformers 5 要求 mapping），hash 变了 | 无害（502 B），不改 Laya 源码，只在 troubleshooting 说明 |
| 5 | 担心 `laya.load` 只认 hub ID | 读源码确认 `Agent.__init__` 先 `os.path.exists()`，本地优先 | 直接 `laya.load("<绝对路径>")`，**不需要兼容层** |
| 6 | 需要证明没走 Hugging Face | — | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 下 load 21.4 s / predict 149 ms 正常；机器上根本没有 `~/.cache/huggingface` |
| 7 | `noul` 概率离谱 | checkpoint 出厂未校准 + zero-shot typed-decisions ≈ 随机 | 不修，作为实验结论写进 README；后续在自己数据上 fine-tune + 拟合温度 |
| 8 | `score` 直接读成 1–5 会错 | 返回值是 0-based 期望值 | `examples/` 同时打印期望值和 `argmax+1` 等级 |
| 9 | WSL 有 GPU 但程序报 CPU | 装了 `+cpu` torch（刻意的） | README/troubleshooting 说明如何切 CUDA，且明确 CPU 才是本阶段目标 |

另外：本项目全程 **没有安装 Ollama**，`pyproject.toml` 里也没有任何生成式模型依赖。

## 8. 验收对照

Environment ☑ WSL 可运行 / ☑ Python 3.11.15 / ☑ uv 0.11.6 / ☑ `uv.lock` 已提交
Model ☑ ModelScope 可下载 / ☑ 落在 `models/laya-multilingual` / ☑ `model.safetensors` 存在且
hash 校验 / ☑ `tokenizer/` 存在 · Laya ☑ `import laya` / ☑ 从本地目录加载 / ☑ 不依赖 Ollama
Inference ☑ 中文输入 / ☑ choice / ☑ score / ☑ noul / ☑ 概率 + confidence
Test ☑ 13 passed / ☑ benchmark 可运行 · Documentation ☑ README / ☑ ModelScope 说明 /
☑ Ollama 说明 / ☑ 架构图 / ☑ troubleshooting

## 9. 下一步（本阶段不实现）

顺序建议，每一步都依赖前一步已经稳定：

1. **Confidence gate**：给 `make demo` 加阈值分支（`conf >= τ` 自动执行，否则交人工）。
   先在自己标好的 200 条中文工单上拟合温度，否则 τ 没有意义（当前 ECE ≈ 0.3）。
2. **接 Ollama（System 2）**：Laya 出决策 → 只有低置信/`other` 时才把 state 交给
   Ollama（qwen 之类）做规划和解释；Ollama 不负责路由判断。分工图见
   `docs/architecture.md`。
3. **Fine-tune `noul` / `score`**：官方 notebook（RLCD + 温度拟合）是唯一能让这两个 primitive
   变可用的路径；不要指望改 prompt 措辞。
4. **MCP / Tools**：Laya 的 `choice` 结果作为「调哪个工具」的路由信号，工具调用由 Code 侧
   执行；MCP server 属于 Code，不属于模型侧。
5. **Router**：出现中英混合流量时再换 `laya.Router(preload=True)`，注意 preload，
   否则每次语言切换重建模型（官方 CPU reload 中位数 7.4 s）。
