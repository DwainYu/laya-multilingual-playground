# Architecture

## 本阶段（已实现）

```text
        WSL2
          │
          ↓
     uv (.venv, Python 3.11)
          │
          ↓
   ModelScope snapshot_download
          │
          ↓
   models/laya-multilingual/         ← 本地目录，唯一权重来源
     model.safetensors · rl_agent_config.json · encoder/ · tokenizer/
          │
          ↓
   laya.load("<local path>")         ← 路径存在 → 完全不碰 hub
          │
          ↓
   agent.predict(state, questions)   ← 一次 forward pass
          │
          ↓
   {answers: {choice | score | noul}, probabilities, confidence}
```

只有这一条链路。没有服务、没有数据库、没有队列。

`src/laya_play/__init__.py` 是唯一共享模块，负责：路径解析、逐文件完整性检查、
`load_agent()`、`describe_runtime()`、`predict_timed()`、结果打印。
`examples/` 与 `tests/` 都只依赖它，不重复实现加载逻辑。

---

## 下一阶段（未实现，只画位置）

```text
                     User
                       │
                       ↓
                    Agent
                       │
              ┌────────┴────────┐
              ↓                 ↓
            Ollama             Laya
              │                 │
       Reasoning/Plan      Decision/Gate
              │                 │
       Explanation /       choice / score / noul
       candidate tools     + probability
              │                 │
              └────────┬────────┘
                       ↓
                     Tools
                       │
                  MCP / HTTP
```

分工是固定的，不要互换：

| 组件 | 职责 | 不负责 |
|---|---|---|
| **Ollama** | 生成式推理、规划、写解释、把决策翻译成人话 | 高频路由判断（太慢、太贵） |
| **Laya** | 快速结构化决策：分类、打分、是否成立 + 概率 | 生成文本、多步规划 |
| **Code** | 执行副作用：调用工具、写库、发请求 | 语义判断 |

接入点已经预留好，不需要改本仓库的代码：

1. **Gate（最先能做的）**：`make demo` 的 `confidence` 字段就是闸门。
   `conf >= 阈值 → 直接执行`；`conf < 阈值 → 才交给 Ollama 推理`。
   System 1 快路径 + System 2 慢路径，成本只花在 Laya 不确定的那部分请求上。
   注意：当前 checkpoint 未校准，阈值必须先用自己标注的数据拟合温度之后再定。
2. **Router**：`laya.Router` 已经内置，按输入文字自动在 english / multilingual /
   typed-decisions 三个 checkpoint 之间选。多语言混合流量时用它，而不是直接 `laya.load`。
   生产要 `Router(preload=True)`，否则每次切语言会重建模型（官方测的 reload 中位数 7.4 s / CPU）。
3. **MCP / Tools**：Laya 只输出「该调哪个工具」这类 `choice`，实际调用由 Code 做。
   MCP server 属于 Code 那一侧，不是 Laya 侧。
4. **Fine-tune**：官方明确 `laya-multilingual` 在 typed-decisions 上 zero-shot ≈ 随机。
   要让 `score` / `noul` 可用，方向是在自己数据上 fine-tune + 拟合温度，
   而不是换 prompt 措辞碰运气。

在加这些东西之前，本阶段的闭环必须先稳定跑通：

```text
ModelScope → Laya → Chinese Decision
```
