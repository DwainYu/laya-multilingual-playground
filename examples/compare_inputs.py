#!/usr/bin/env python3
"""Ten Chinese scenarios through the same typed questions: decision, confidence, latency.

    uv run python examples/compare_inputs.py
"""

from __future__ import annotations

from laya_play import banner, describe_runtime, load_agent, predict_timed
from laya_play.questions import DEPARTMENT_CRITERIA, TYPED_QUESTIONS

CASES = [
    ("重复扣款", "我的订单昨天被扣了两次钱，希望尽快退款。"),
    ("网络无法连接", "连不上公司网络，客户端一直提示连接超时，已经持续两个小时了。"),
    ("AI 模型启动失败", "我们部署的大模型服务启动失败，日志显示模型权重文件加载异常。"),
    ("页面打不开", "官网页面一直转圈打不开，换浏览器刷新多次也没有用。"),
    ("订单物流延迟", "订单显示已发货，但三天没有任何物流更新，请问包裹到哪里了？"),
    ("密码无法登录", "我修改密码之后无法登录，一直提示账号或密码错误。"),
    ("请求退款", "商品还没有发货，我想取消订单并申请全额退款。"),
    ("模型推理速度很慢", "模型推理接口每次要等十几秒，把 batch size 调小也没有改善。"),
    ("GPU 显存不足", "训练时报 CUDA out of memory，单卡 24G 显存不够用，怎么优化？"),
    ("普通产品咨询", "请问你们的产品支持哪些语言？有没有企业版和私有化部署？"),
]


def main() -> int:
    banner("Compare Chinese inputs")

    agent = load_agent()
    print("Runtime:")
    print(describe_runtime(agent))

    print("\nQuestions per request: department [choice] + urgency [score] + "
          "ai_deployment_related [noul]\n")

    rows = []
    for title, text in CASES:
        state = {"subject": title, "body": text}
        result, ms = predict_timed(agent, state, TYPED_QUESTIONS)
        answers = result["answers"]
        department = answers["department"]
        rows.append(
            {
                "title": title,
                "text": text,
                "department": department["choice"],
                "confidence": department["confidence"],
                "probabilities": department["probabilities"],
                "urgency": answers["urgency"]["score"],
                "ai": answers["ai_deployment_related"]["noul"],
                "ms": ms,
            }
        )

    width = max(len(r["text"]) for r in rows)
    print(f"{'input':<{width}}  {'decision':<10}{'conf':>6}{'urgency':>9}{'ai?':>7}{'ms':>8}")
    print("-" * (width + 42))
    for row in rows:
        print(f"{row['text']:<{width}}  {row['department']:<10}{row['confidence']:>6.3f}"
              f"{row['urgency']:>9.2f}{row['ai']:>7.2f}{row['ms']:>8.1f}")

    print("\nPer-input probability detail:")
    labels = list(DEPARTMENT_CRITERIA)
    for row in rows:
        probs = " ".join(f"{label}={row['probabilities'][label]:.3f}" for label in labels)
        print(f"\n[{row['title']}] {row['text']}\n  {probs}")

    latencies = [row["ms"] for row in rows]
    avg = sum(latencies) / len(latencies)
    print(f"\nLatency over {len(latencies)} requests: "
          f"min {min(latencies):.1f} ms / avg {avg:.1f} ms / max {max(latencies):.1f} ms")
    print("=" * 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
