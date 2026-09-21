# Troubleshooting

本项目实际踩过的坑都在下面，每条都是真实报错 + 真实修法。末尾是通用诊断顺序。

---

## 加载 / 安装

### `uv sync` 失败：`OSError: Readme file does not exist: README.md`

```text
File ".../hatchling/metadata/core.py", line 553, in readme
    raise OSError(message)
OSError: Readme file does not exist: README.md
```

`pyproject.toml` 里声明了 `readme = "README.md"`，而本项目自身会被 hatchling 装成
editable 包（`src/laya_play`）。README.md 不存在时 build backend 直接失败。

**修法**：先有 README.md 再 `uv sync`。本仓库已包含，正常 clone 不会遇到。

### `laya.load()` 卡住不动

`transformers` 在 import 时探测 TensorFlow；如果环境里装了 TF，其 abseil runtime 可能
deadlock 模型构建（Laya 官方 README 明确提到）。

**修法**：在 **import laya 之前** 设置 `USE_TF=0`。

```python
import os
os.environ.setdefault("USE_TF", "0")   # 顺序错了就没用
import laya
```

`src/laya_play/__init__.py` 顶部已经这么做了，`Makefile` 里 `export USE_TF := 0`。

本环境未安装 TensorFlow，所以这里只是预防。**注意设晚于 import 是无效的**，这是最常见的
错误写法。

### 加载要 ~20 秒，不是死锁

CPU 冷启动实测 `laya.load()` ≈ 20–25 s：322M 参数以 float32 读入 + 256k 词表 tokenizer
反序列化（tokenizer.json 33 MiB）。**这是正常耗时**，warm 之后每次 predict 只有 ~130 ms。
只有当超过 1–2 分钟没输出时才按上面的 USE_TF 排查。

---

## ModelScope 下载

### 第二次 `make download` 会重新拉 tokenizer_config.json

现象：权重不重下，但

```text
Downloading: 100%|██████████| 8/8 [...]
tokenizer_config.json: 0.00/502 [...]
```

原因：`laya.load()` 会**就地改写** `models/laya-multilingual/tokenizer/tokenizer_config.json`
（`laya/agent.py::_fix_tokenizer_config`）：把 `tokenizer_class` 归一化、并把
`extra_special_tokens` 从 list 转成 dict（这个 checkpoint 的该字段是 list，
transformers 5 要求 mapping，否则 `AutoTokenizer` 直接抛
`'list' object has no attribute 'keys'`）。文件 hash 一变，ModelScope 就认为需要重下。

**结论**：无害。该文件只有 502 B，重下不到 1 s。不要为此改 Laya 源码，也不要把
`models/` 纳入 Git。

### `snapshot_download` 参数

本项目用 `local_dir=`，把文件直接落到 `models/laya-multilingual/`，而不是全局缓存
（`~/.cache/modelscope/hub`）。modelscope 1.40.1 的 `snapshot_download(MODEL_ID, local_dir=...)`
返回值就是那个目录的绝对路径，脚本会打印出来核对。

### 完整性检查必须是逐文件

`os.path.isdir("models/laya-multilingual")` 为真但只有半套权重是常见状态（下载中断）。
所以 `scripts/download_model.py` 与 `laya_play.missing_files()` 逐个检查：

```text
README.md · model.safetensors · rl_agent_config.json
encoder/config.json · tokenizer/tokenizer.json · tokenizer/tokenizer_config.json
```

缺任何一个 → 打印缺失清单并以 **exit 1** 结束。smoke test 里也是逐文件断言。

### 走 Hugging Face 的 fallback

只有当 ModelScope 拉不下来时才需要。删掉 `models/laya-multilingual`，然后：

```python
agent = laya.load("convaiinnovations/laya-multilingual")   # 走 huggingface_hub
```

注意这条路径会写 `~/.cache/huggingface`，且国内直连通常不通，需要 `HF_ENDPOINT` 镜像。
本项目的验收路径是 ModelScope，不依赖 HF：`HF_HUB_OFFLINE=1` 下加载推理全部正常。

---

## torch / 设备

### `cuda available: no`，但有 GPU

WSL2 里 `nvidia-smi` 正常（RTX 3060 Laptop，driver 566.07）而程序仍报 CPU，是因为
本项目刻意装的是 **CPU 版 torch**：

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

这样避免为一个 CPU 实验拖 2 GB+ 的 CUDA wheel 和 nvidia-* 依赖。**这是设计选择，不是故障**：
第一阶段目标就是 `WSL + CPU` 能跑完 smoke test（README「Local WSL benchmark」）。

要用 GPU，把 torch 换成 CUDA 版后 `uv sync`：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 'torch>=2.0.0'
agent = laya.load("models/laya-multilingual", device="cuda")   # 或 laya.load(..., device="cuda")
```

模型 fp16 权重 ~614 MiB，6 GB 显存够用。届时 benchmark 数字要重新测，
不能引用官方 T4 的 32.8 ms。

### 别把官方 benchmark 当本地 benchmark

官方 32.8 ms / 1 question 是 **Tesla T4 GPU**。本机 CPU 是 ~129 ms。README 里两块是分开的。

---

## 结果解读

### `noul` / `score` 的概率不能直接用

实测：问「这个请求是否与 AI 模型部署有关？」，一条纯退款工单返回 `noul = 0.9409`，
「页面打不开」返回 0.97 —— 明显错，而且非常自信。

不是本项目 bug。官方模型卡写明：

- `laya-multilingual` 在 typed-decisions 上 zero-shot 0.342，随机基线 0.318，多数类基线 0.461；
- 出厂 `temperature = [1.0, 1.0, 1.0]`，**未校准**，systematically over-confident
  （平均 confidence 0.75–0.83 对应远低于此的 accuracy）；
- ordinal `score` 是最弱的 primitive（SST-5 0.282）。

**要能用**：在自己的标注数据上 fine-tune，再按 (question type, option count) 拟合温度
（官方数据：ECE 0.314 → 0.106）。在此之前只把 `choice` 当可用能力。

### `score` 是 0-based 期望值，不是等级

`urgency` 给了 5 级 rubric，返回的 `score` 是 `Σ i·p(i)`，范围 `0..4`，不是 `1..5`。
要显示成第几级：用 `probabilities` 的 argmax 再 +1，或直接 `round(score) + 1`。
`examples/typed_decisions.py` 里两样都打印了。

### `choice` 选项数

`laya-multilingual` 的 `head_max_len = 256`（`rl_agent_config.json`），所有选项共享这个预算。
选项超过 ~20 个时每个标签只剩几个 token，准确率会崩（官方 Banking77 0.425）。
本项目 5 个选项，安全。真要放大标签空间：改 `agent.cfg["head_max_len"]` / `max_len`，
或拆成 coarse-to-fine 两级 choice。若超限，Laya 会直接抛
`question 'x' options exceed head_max_len=256`，而不是静默给错答案。

### 概率不写进断言

`tests/test_smoke.py` 只断言结构（字段存在、标签合法、概率求和 ≈ 1、数值在 0–1 之间）。
**不要**加 `assert probs["billing"] == 0.97` 这类断言：checkpoint 未校准，数值会随版本浮动。

---

## 诊断顺序

任何一步失败（`uv sync` / download / import laya / load / predict），按这个顺序查，
别停在第一行错误信息上：

```bash
make setup && uv run python - <<'PY'
import os, platform, sys
print("1 python      ", platform.python_version(), sys.executable)
print("0 uv          ", os.popen("uv --version").read().strip())
import laya, torch, transformers, safetensors, modelscope
print("2 laya        ", laya.__version__)
print("3 torch       ", torch.__version__, "cuda:", torch.cuda.is_available(),
      "threads:", torch.get_num_threads())
print("4 transformers", transformers.__version__)
print("5 safetensors ", safetensors.__version__)
print("6 modelscope  ", modelscope.__version__)
print("7 USE_TF      ", repr(os.environ.get("USE_TF")), "(必须是 '0'，且必须在 import laya 前)")
from laya_play import model_dir, missing_files
print("8 model dir   ", model_dir(), "missing:", missing_files())
print("9 laya.load   ", __import__("inspect").signature(laya.load))
PY
```

逐项对应的真实坑：

| # | 看什么 | 出问题通常是 |
|---|---|---|
| 1 | Python ≥ 3.11 且来自 `.venv` | 用了系统 python 3.10；忘了 `source .venv/bin/activate` |
| 2 | laya 0.3.x | 装到同名的无关包（确认 `laya.__version__` 且 author 是 Convai） |
| 3 | `2.14.0+cpu` | 装了 CUDA wheel 但驱动不匹配；或反之 |
| 4 | transformers 5.x | checkpoint 的 `encoder/config.json` 写于 5.0.0，4.x 可能读不到新字段 |
| 5–6 | safetensors / modelscope | 版本过旧导致 `local_dir=` 行为不同 |
| 7 | `USE_TF=0` | 设在 import 之后 → 等于没设 |
| 8 | `missing_files()` | 下载中断，目录存在但半套权重 |
| 9 | 签名 `(model_id_or_path, device, token, subfolder)` | 传了不存在的路径 → 它会去 HF 下载 |

最小重现（把问题压到一次调用）：

```bash
export USE_TF=0
uv run python -c "
from laya_play import load_agent
a = load_agent()
print(a.predict({'body':'我的订单昨天被扣了两次钱，希望尽快退款。'},
                {'d':{'type':'choice','instructions':'交给哪个部门？',
                      'criteria':{'billing':'扣款退款','technical':'故障报错'}}}))
"
```

这条能跑通，就说明 ModelScope → 本地目录 → Laya → 中文决策 整条链路没问题，
剩下的都是上层示例的问题。
