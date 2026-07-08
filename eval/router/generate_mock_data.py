"""Generate synthetic annotated routing data for testing the ACE pipeline.

Produces ~300 samples across all 6 agents, with both A-class (badcase + human_note)
and B-class (correct prediction) data.
"""

import json
import random
import os

# All 6 agent names
AGENTS = [
    "逾期及APP使用",
    "还款",
    "协商还款咨询",
    "借款/会员Agent",
    "提前结清咨询",
    "贷款申请及优惠券使用",
]

# Agent-specific query templates with correct agent
QUERY_TEMPLATES = {
    "逾期及APP使用": [
        # 逾期/征信/催收
        ("我逾期了会有什么影响", "咨询逾期影响"),
        ("上征信了怎么办能删除吗", "咨询更改征信相关"),
        ("帮我查下现在有没有逾期", "查询当前是否逾期"),
        ("催收什么时候会打电话给我", "咨询催收规则"),
        ("能不能帮我办停催", "办理停止催收"),
        ("想注销账户怎么操作", "办理注销账户相关"),
        ("怎么修改注册手机号", "咨询如何更改注册手机号"),
        ("APP登录密码忘了怎么办", "咨询登录APP异常"),
        ("在哪里看商城物流信息", "查询商城物流信息"),
        ("怎么改紧急联系人", "咨询如何查看及修改紧急联系人"),
        ("APP登录收不到短信验证码", "反馈登录时收不到短信验证码"),
        ("账户被冻结了怎么恢复", "咨询如何恢复账户"),
        ("怎么退出当前登录的账户", "咨询如何退出账户"),
        ("未借款但收到短信提醒", "咨询未借款收到短信"),
        ("微信公众号怎么关注", "咨询微信公众号相关"),
        ("这是不是虚假APP", "质疑虚假APP相关特殊情景服务咨询"),
        ("想知道注册时间是什么时候", "查询注册时间"),
        ("逾期罚息怎么算", "咨询逾期罚息"),
        ("给我开个非恶意逾期证明", "开具非恶意逾期证明相关"),
        ("怎么修改交易密码", "咨询如何修改交易密码"),
    ],
    "还款": [
        ("我还剩多少钱没还", "查询剩余欠款总额"),
        ("怎么查历史还款记录", "查询还款记录"),
        ("扣款失败了怎么回事", "咨询还款失败原因"),
        ("已经还了还显示欠款", "咨询已还款还显示欠款相关"),
        ("扣款规则是什么", "咨询扣款规则"),
        ("怎么开发票", "咨询如何开具发票相关问题"),
        ("还款日能不能改", "查询或更改还款日相关"),
        ("支持哪些方式还款", "咨询如何快速还款相关"),
        ("银行卡被冻结了还款怎么办", "咨询银行卡冻结"),
        ("怎么看合同是否结清了", "查询合同是否结清"),
        ("扣了我两次钱怎么办", "咨询重复还款原因及溢缴款提取"),
        ("怎么开结清证明", "咨询如何开具结清证明"),
        ("利息怎么算的", "咨询贷款利息"),
        ("本期应还多少钱", "咨询本期应还金额"),
        ("最低还款额是多少", "咨询最低还款额"),
        ("帮我改绑定银行卡", "咨询更改或绑定银行卡相关"),
        ("想支付宝还款怎么操作", "咨询使用支付宝还款相关"),
        ("微信还款怎么还", "咨询使用微信还款相关"),
        ("帮我查还款成功了没", "查询还款是否成功"),
        ("想对公转账还款", "咨询如何线下对公还款相关"),
    ],
    "协商还款咨询": [
        ("最近资金困难能不能延期还款", "协商还款方案咨询"),
        ("想协商减免一些利息", "协商减免还款方案咨询"),
        ("帮我重新分期可以吗", "咨询再分期相关"),
        ("能不能只还本金不还利息", "询问只还本金相关"),
        ("展期是什么意思可以办吗", "咨询展期业务相关"),
        ("我想把消费转成分期", "咨询消费转分期"),
        ("分期的期数能改吗", "咨询更改期数相关"),
        ("想关闭再分期功能", "询问关闭再分期相关"),
        ("延期还款怎么申请", "协商延期还款咨询"),
        ("实在还不上了有什么方案", "协商还款方案咨询"),
    ],
    "借款/会员Agent": [
        ("我额度多少", "查询额度相关"),
        ("为什么提现失败了", "咨询提现失败原因"),
        ("额度怎么降了", "咨询额度降低原因"),
        ("借款什么时候到账", "咨询借款到账时间"),
        ("想取消这次借款", "办理取消借款"),
        ("会员自动续费怎么关", "咨询如何操作关闭会员续费"),
        ("帮我查下会员状态", "查询会员状态"),
        ("会员费能退吗", "咨询会员退费"),
        ("极速放款是什么", "咨询极速放款权益"),
        ("额度不能用怎么回事", "咨询额度无法使用原因"),
        ("提额失败了什么原因", "咨询提额失败原因"),
        ("怎么看借款记录", "查询借款记录"),
        ("额度有效期多久", "咨询额度有效期"),
        ("会员有什么权益", "咨询会员权益"),
        ("临时额度能借吗", "咨询如何使用固定或者临时额度"),
        ("借款用途有什么限制", "咨询借款用途"),
        ("额度冻结了怎么解", "咨询额度被冻结相关"),
        ("怎么关闭额度", "关闭额度"),
        ("这个小马花花卡是什么", "咨询小马花花卡"),
        ("到账金额跟我借的不一样", "反馈借款到账金额问题"),
    ],
    "提前结清咨询": [
        ("想一次性还清所有欠款", "咨询提前结清规则及办理路径"),
        ("提前还清利息怎么算", "咨询提前结清利息"),
        ("怎么操作提前结清", "咨询提前结清规则及办理路径"),
        ("提前结清失败了为什么", "咨询提前结清失败原因"),
        ("提前还款预约的日期能改吗", "咨询提前还款预约日期查询和修改"),
        ("分了期还能提前还清吗", "咨询再分期后是否可提前结清"),
        ("提前结清手续费多少", "咨询提前结清手续费"),
        ("转错了金额能退吗", "咨询退还借错款项"),
        ("结清按期还款有优惠券吗", "咨询结清按期还款优惠券相关"),
        ("想全部还清APP里没找到入口", "咨询提前结清规则及办理路径"),
    ],
    "贷款申请及优惠券使用": [
        ("怎么申请贷款", "咨询如何贷款"),
        ("借款需要什么条件", "咨询借款资质要求"),
        ("能分多少期", "咨询产品可分期期数"),
        ("上次没还完还能借吗", "咨询借款没还完是否还能再次借款"),
        ("怎么申请提额", "咨询如何申请提额"),
        ("征信不好能借吗", "咨询征信不足相关"),
        ("什么时候能再借款", "咨询可借款时间"),
        ("优惠券在哪里看怎么用", "咨询优惠券查询及使用"),
        ("借款会上征信吗", "咨询借款是否上报征信"),
        ("循环额度什么意思", "咨询循环额度"),
        ("人脸识别一直过不了", "咨询人脸识别不通过怎么办"),
        ("申请临时额度失败了", "咨询提取临时额度失败原因"),
        ("借款要准备什么资料", "咨询借款资料相关"),
        ("怎么授权公积金提额", "咨询公积金授权提额相关"),
        ("预约借款能保证成功吗", "咨询借款失败时的规则解释"),
        ("临时额度是什么", "咨询临时额度是什么"),
        ("收不到验证码怎么申请借款", "反馈借款时收不到短信验证码"),
        ("逾期了还能申请借款吗", "咨询逾期后申请借款相关"),
        ("为什么要人脸识别", "咨询为什么要人脸识别"),
        ("能线下借款吗", "咨询线下进行贷款"),
    ],
}

# Common dialogue history scenarios
HISTORY_SCENARIOS = [
    # 还款相关历史
    [
        "user: 你好，我想问下还款的事",
        "assistant: 您好，请问您想了解哪方面的还款问题呢？"
    ],
    [
        "user: 我的账单出来了，想问下怎么还",
        "assistant: 您好，您可以通过APP首页的还款入口进行还款，支持支付宝、微信等多种方式。需要我帮您查看具体账单金额吗？"
    ],
    # 逾期相关历史
    [
        "user: 我收到一条短信说我逾期了",
        "assistant: 您好，我来帮您确认一下。请问您能提供一下您的注册手机号吗？"
    ],
    # 借款相关历史
    [
        "user: 我想借点钱",
        "assistant: 您好，您可以在APP首页点击借款入口，系统会根据您的额度情况展示可借金额。需要我帮您查看当前额度吗？"
    ],
    [
        "user: 上次借款被拒了，想问下原因",
        "assistant: 您好，借款审核结果取决于多方面的因素，包括征信情况、还款记录等。我可以帮您查看具体原因。"
    ],
    # 会员相关历史
    [
        "user: 我好像开通了什么会员",
        "assistant: 您好，请问是逸骊生活会员吗？我可以帮您查询会员状态。"
    ],
    # 结清相关历史
    [
        "user: 我想把欠款全部还了",
        "assistant: 您好，您是想办理提前结清吗？请问您需要了解提前结清的规则还是直接办理？"
    ],
    # 协商相关历史
    [
        "user: 最近压力很大，还款有点困难",
        "assistant: 我理解您的情况。我们有一些灵活的还款方案可以帮您缓解压力，需要我帮您介绍一下吗？"
    ],
    # 无历史
    [],
]

# Common router_reason templates for badcase (wrong agent chosen)
BAD_ROUTER_REASONS = {
    "上下文误导": "【测试组】根据对话历史，用户此前在咨询{history_topic}相关问题，当前query虽然涉及不同意图，但从对话连贯性来看应继续由{history_agent}处理。",
    "关键词松散匹配": "【测试组】query中提到的关键词与Agent职责描述中的术语存在语义关联，匹配到{related_agent}的职责范围，因此路由到{related_agent}。",
    "多步联想": "【测试组】从query的表层含义出发，通过概念联想推测用户可能最终目的与{target_agent}相关，选择了该Agent。",
}

# Human notes for badcase — the KEY training signal
HUMAN_NOTES = {
    "逾期及APP使用": [
        "query中明确提到'{query_keyword}'，这是逾期及APP使用Agent的具体意图，应直接匹配",
        "这属于APP操作类问题，与当前对话历史中的还款话题无关",
        "用户问的是账户管理操作，按操作归因优先原则属于逾期及APP使用",
        "更改个人信息/账号设置类操作，不管对话在聊什么都归逾期及APP使用",
    ],
    "还款": [
        "query在问还款相关操作，对话历史虽然聊别的但用户话题显然回到了还款",
        "用户关心的是当期账单/还款操作，属于还款Agent核心职责",
        "虽然对话历史涉及其他业务，但当前query明确表达还款意图",
    ],
    "协商还款咨询": [
        "用户表达了资金困难和希望协商的意图，这是协商还款咨询的职责",
        "延期/减免/再分期这些关键词匹配到协商还款咨询Agent",
    ],
    "借款/会员Agent": [
        "额度/提现/会员相关问题，属于借款/会员Agent职责范围",
        "这是会员业务范畴的操作，不是还款也不是贷款申请",
        "用户关心的是借款操作/额度使用，属于借款/会员Agent",
    ],
    "提前结清咨询": [
        "关键词'提前还清'/'一次性'明确指向提前结清咨询",
        "这是一次性全部还款的意图，不是当期账单还款",
    ],
    "贷款申请及优惠券使用": [
        "用户在咨询贷款申请流程，属于贷款申请Agent职责",
        "优惠券查询和使用问题，是贷款申请及优惠券Agent的专属职责",
    ],
}


def generate_queries_for_agent(agent, n=30):
    """Generate queries for a specific agent."""
    templates = QUERY_TEMPLATES[agent]
    samples = []
    for _ in range(n):
        query, intent = random.choice(templates)
        # Add some natural variation
        variations = [
            query,
            f"你好，{query}",
            f"问一下，{query}",
            f"请问{query}",
            f"{query}？",
        ]
        query_text = random.choice(variations)
        samples.append({"query": query_text, "intent": intent, "correct_agent": agent})
    return samples


def pick_wrong_agent(correct_agent):
    """Pick a plausibly-confusable wrong agent."""
    confusion_pairs = {
        "逾期及APP使用": ["还款", "借款/会员Agent"],
        "还款": ["逾期及APP使用", "提前结清咨询", "借款/会员Agent"],
        "协商还款咨询": ["还款", "提前结清咨询"],
        "借款/会员Agent": ["贷款申请及优惠券使用", "还款"],
        "提前结清咨询": ["还款", "协商还款咨询"],
        "贷款申请及优惠券使用": ["借款/会员Agent", "逾期及APP使用"],
    }
    candidates = confusion_pairs.get(correct_agent, [])
    if candidates:
        return random.choice(candidates)
    return random.choice([a for a in AGENTS if a != correct_agent])


def generate_bad_reason(wrong_agent, correct_agent, history):
    """Generate a plausible but wrong router_reason."""
    template_key = random.choice(list(BAD_ROUTER_REASONS.keys()))
    template = BAD_ROUTER_REASONS[template_key]
    history_agent = "还款" if "还款" in str(history) else random.choice(AGENTS)
    history_topic = random.choice(["还款", "借款", "会员", "逾期", "催收", "额度"])

    if template_key == "上下文误导":
        return template.format(history_topic=history_topic, history_agent=history_agent)
    elif template_key == "关键词松散匹配":
        return template.format(related_agent=wrong_agent)
    else:
        return template.format(target_agent=wrong_agent)


def generate_human_note(correct_agent, query):
    """Generate a concise human annotation note."""
    notes = HUMAN_NOTES.get(correct_agent, [f"正确应为{correct_agent}"])
    note_template = random.choice(notes)
    # Extract a keyword from query for the note
    query_keyword = query[:15]
    return note_template.format(query_keyword=query_keyword)


def generate_mock_data(n_total=350, badcase_ratio=0.25):
    """
    Generate synthetic annotated data.

    Args:
        n_total: Total samples (~350)
        badcase_ratio: ~25% are badcases (A-class with human annotation)

    Returns:
        List of data dicts
    """
    samples = []

    # Generate balanced samples across agents
    per_agent = n_total // len(AGENTS)  # ~58 per agent
    all_queries = []
    for agent in AGENTS:
        queries = generate_queries_for_agent(agent, n=per_agent)
        all_queries.extend(queries)

    random.shuffle(all_queries)

    for i, q in enumerate(all_queries):
        correct_agent = q["correct_agent"]
        query = q["query"]
        history = random.choice(HISTORY_SCENARIOS)

        # Determine if this is a badcase
        is_badcase = random.random() < badcase_ratio

        if is_badcase:
            wrong_agent = pick_wrong_agent(correct_agent)
            # Ensure wrong != correct
            attempts = 0
            while wrong_agent == correct_agent and attempts < 10:
                wrong_agent = pick_wrong_agent(correct_agent)
                attempts += 1
            if wrong_agent == correct_agent:
                is_badcase = False

        if is_badcase:
            router_agent = wrong_agent
            router_reason = generate_bad_reason(wrong_agent, correct_agent, history)
            human_note = generate_human_note(correct_agent, query)
            human_annotate_agent = correct_agent
            final_agent = correct_agent
        else:
            router_agent = correct_agent
            router_reason = f"【线上路由】query中的意图关键词与Agent职责中的具体意图匹配，路由到{correct_agent}。"
            human_note = ""
            human_annotate_agent = ""
            final_agent = correct_agent

        sample = {
            "session_id": f"mock_{i:05d}",
            "source": "synthetic_test_data",
            "timestamp": f"2026-07-{random.randint(1,28):02d} {random.randint(8,22):02d}:{random.randint(0,59):02d}:{random.randint(0,59):02d}",
            "historyDialogue": history,
            "query": query,
            "current_agent": random.choice(["闲聊"] + AGENTS),
            "router_reason": router_reason,
            "router_agent": router_agent,
            "human_note": human_note,
            "human_annotate_agent": human_annotate_agent,
            "final_agent": final_agent,
        }
        samples.append(sample)

    # Shuffle one more time
    random.shuffle(samples)

    # Print stats
    a_count = sum(1 for s in samples if s["human_annotate_agent"])
    b_count = len(samples) - a_count
    print(f"Generated {len(samples)} samples: {a_count} A-class (badcase), {b_count} B-class (correct)")

    # Per-agent distribution
    from collections import Counter
    agent_dist = Counter(s["final_agent"] for s in samples)
    for agent, count in agent_dist.most_common():
        a_in_agent = sum(1 for s in samples if s["final_agent"] == agent and s["human_annotate_agent"])
        print(f"  {agent}: {count} total, {a_in_agent} badcase")

    return samples


if __name__ == "__main__":
    random.seed(42)
    samples = generate_mock_data(n_total=350, badcase_ratio=0.25)

    output_path = os.path.join(os.path.dirname(__file__), "data", "mock_annotated.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

    print(f"\nSaved to {output_path}")
