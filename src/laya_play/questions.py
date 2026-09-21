"""Chinese support-ticket state and typed question schemas shared by the examples."""

from __future__ import annotations

DEPARTMENT_CRITERIA = {
    "billing": "账单、扣款、退款、发票",
    "technical": "报错、崩溃、故障、无法运行",
    "logistics": "发货、快递、配送、物流延迟",
    "account": "账号、登录、密码、权限",
    "other": "咨询、介绍、不属于以上任何一类",
}

URGENCY_LEVELS = [
    "不紧急",
    "一般",
    "紧急",
    "严重影响业务",
    "紧急故障",
]

TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "这个工单应该交给哪个部门处理？",
        "criteria": DEPARTMENT_CRITERIA,
    },
}

TYPED_QUESTIONS = {
    "department": TRIAGE_QUESTIONS["department"],
    "urgency": {
        "type": "score",
        "instructions": "这条请求有多紧急？",
        "criteria": URGENCY_LEVELS,
    },
    "ai_deployment_related": {
        "type": "noul",
        "instructions": "这个请求是否与 AI 模型部署有关？",
        "criteria": {
            "false": "与模型部署、推理、显存、训练无关",
            "true": "涉及模型部署、推理服务、显存或训练",
        },
    },
}

REFUND_TICKET = {
    "subject": "订单重复扣款",
    "body": "我的订单昨天被扣了两次钱，希望尽快退款。",
}
