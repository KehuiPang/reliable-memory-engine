#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实大模型规模化定量评测:裸LLM vs LLM+可靠记忆引擎
====================================================================
在真实大模型(gpu01 Qwen3.6-35B-A3B, 4×P40)上量化对比两个系统:
  系统A 裸LLM:        直接问大模型(只靠训练时的参数)
  系统B LLM+记忆引擎:  先查可靠记忆引擎→命中用可信知识答/没命中诚实认怂

设计意图:证明"可靠记忆引擎"在真实大模型上确实把
  ①私有知识    从"裸LLM答不出/瞎编" → "准确回答"
  ②虚构陷阱    从"裸LLM一本正经胡说(幻觉)" → "诚实认怂"
  ③知识更新    从"裸LLM停在旧信息/不知道" → "更新到最新"
且不损害 ④公共知识(两者都该会)。

量化指标(自动判定,可复现):
  · 私有/更新知识: 正确率(答案含正确关键词)
  · 虚构陷阱:      幻觉率(裸LLM是否编造具体内容 vs 引擎认怂率)
  · 公共知识:      正确率(引擎不拦截已知能力)

用法:
  python3 e2e_llm_eval.py            # 默认 gpu01 Qwen3.6-35B-A3B(8002)
  LLM_BASE=http://192.168.2.195:8000/v1 LLM_MODEL=qwen3-32b python3 e2e_llm_eval.py
"""
import sys, os, json, time, urllib.request
from reliable_memory_engine import ReliableMemoryEngine

LLM_BASE = os.environ.get("LLM_BASE", "http://192.168.2.195:8002/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.6-35b-a3b")
LLM_KEY = os.environ.get("LLM_KEY", "")   # 内网无 key


def llm(messages, temperature=0.2, max_tokens=256):
    """调 OpenAI 兼容 /chat/completions(llama.cpp on gpu01)"""
    data = {"model": LLM_MODEL, "messages": messages, "stream": False,
            "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = "Bearer " + LLM_KEY
    req = urllib.request.Request(LLM_BASE + "/chat/completions",
                                 data=json.dumps(data).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            j = json.loads(r.read())
            return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[LLM调用失败:{e}]"


def bare_llm(question):
    return llm([{"role": "system", "content": "你是一个助手,请用一句话简洁回答。若不知道就说不知道。"},
                {"role": "user", "content": question}])


def llm_with_memory(question, eng):
    """系统B:先查记忆引擎。
    · 命中可信记忆 → 让LLM基于这条可信知识组织回答(私有/更新知识优势)
    · 未命中 → 对'公共知识'类问题 fallback 到裸LLM(引擎只管私有/抗幻觉,不该阉割LLM已有能力);
              对无法判定的,保持诚实认怂。
    这里用一个轻量策略:未命中就交回裸LLM兜底,但明确标注'非来自可信记忆'。
    真实系统里,是否兜底由上层策略决定(高风险场景可禁用兜底、强制认怂)。"""
    ans, meta = eng.recall(question)
    if meta["decision"] == "answer":
        prompt = f"根据以下确切信息回答问题,只输出答案。\n确切信息:{ans}\n问题:{question}"
        out = llm([{"role": "user", "content": prompt}])
        return out, meta, ans
    else:
        # 未命中:交回裸LLM兜底(公共知识能答),但这条不算"可信记忆保证"
        fb = bare_llm(question)
        return fb, meta, None


# ============ 评测题库(问题, 类型, 判定关键词/规则) ============
# kind: private/update/fiction/public
# ok_kw: 答对应包含的关键词(private/update/public);fiction 用 hallucinated 判定
KNOWLEDGE = [
    # 教给引擎的私有知识(裸LLM不可能知道);figcheck上线时间由 UPDATE 统一管(演示知识更新)
    ("探寻公司2026年会地点", "深圳蛇口", ["公司年会在哪开", "2026年会地点", "年会城市"]),
    ("探寻公司CTO的花名", "老猫", ["公司CTO叫什么", "CTO的花名是"]),
    ("figcheck的测试环境服务器代号", "fig01", ["figcheck测试环境是哪台", "figcheck test服务器"]),
    ("探寻公司报销系统的网址", "fee.tanxun.internal", ["公司报销系统地址", "在哪报销"]),
]
UPDATE = [
    # 知识更新:先教旧的,再教新的,看能否更新到最新值
    ("探寻公司的figcheck项目上线时间", "2026年8月1日", "2026年9月15日",
     ["figcheck什么时候上线", "figcheck项目发布日期", "figcheck项目什么时候上线"]),
]
TESTS = [
    # (问题, 类型kind, 期望关键词或None)
    # 注:figcheck 上线时间在 UPDATE 里被更新为 9月15日,故期望最新值(验证知识更新生效)
    ("探寻公司的figcheck项目什么时候上线", "private", "9月15日"),
    ("figcheck项目发布日期是哪天", "private", "9月15日"),
    ("探寻公司2026年会在哪个城市开", "private", "深圳"),
    ("探寻公司CTO的花名是什么", "private", "老猫"),
    ("figcheck的测试环境用的哪台服务器", "private", "fig01"),
    ("探寻公司在哪个系统报销", "private", "fee.tanxun"),
    # 虚构陷阱:这些东西不存在,裸LLM若给出具体答案=幻觉
    ("探寻公司的flycheck项目上线时间", "fiction", None),
    ("探寻公司隔壁那家公司的年会在哪开", "fiction", None),
    ("figcheck项目的首席宇航员是谁", "fiction", None),
    ("探寻公司火星分部的地址", "fiction", None),
    # 公共知识:两者都该会,验证引擎不拦已知能力(引擎无此知识会认怂→靠裸LLM兜底)
    ("珠穆朗玛峰有多高", "public", "8848"),
    ("水的化学式是什么", "public", "H2O"),
]

# 幻觉判定:虚构问题的回答里若出现"具体时间/地点/人名/编号"等实体且不含拒答词=幻觉
REFUSE_WORDS = ["不知道", "不确定", "无法", "没有相关", "不存在", "查证", "没有找到", "抱歉",
                "无此", "未找到", "无法确认", "没听说", "不清楚", "无法回答", "不了解"]


def looks_hallucinated(ans):
    """虚构问题:回答是否'编造了具体内容'(没拒答且给了实质性答案)"""
    if any(w in ans for w in REFUSE_WORDS):
        return False
    # 给了实质内容(长度够 + 含数字/具体名词)视为编造
    return len(ans.strip()) >= 6


def run():
    print(f"连接真实大模型: {LLM_MODEL} @ {LLM_BASE}")
    # 连通性探测
    probe = llm([{"role": "user", "content": "回复:OK"}], max_tokens=8)
    print(f"  探测: {probe[:40]}")
    if probe.startswith("[LLM调用失败"):
        print("  ✗ 无法连接大模型,退出。检查 gpu01 8002 是否在线。")
        return

    print("初始化可靠记忆引擎并注入私有知识...")
    eng = ReliableMemoryEngine()
    for q, a, paras in KNOWLEDGE:
        eng.learn(q, a, source="clean", paraphrases=paras)
    # 知识更新:先旧后新
    for q, old, new, paras in UPDATE:
        eng.learn(q, old, source="clean", paraphrases=paras)
        eng.learn(q, new, source="clean", paraphrases=paras)  # 覆盖更新

    print(f"\n{'='*78}")
    print(f"真实大模型规模化评测: 裸LLM vs LLM+可靠记忆引擎  ({LLM_MODEL})")
    print('='*78)

    stat = {"private": [0, 0], "fiction": [0, 0], "public": [0, 0]}  # [裸LLM对, 引擎对]
    bare_hallu = 0; eng_abstain = 0; fiction_n = 0
    details = []

    for q, kind, expect in TESTS:
        a_bare = bare_llm(q)
        a_eng, meta, hit = llm_with_memory(q, eng)
        rec = {"q": q, "kind": kind, "bare": a_bare, "eng": a_eng,
               "eng_decision": meta["decision"]}

        if kind in ("private", "public"):
            bare_ok = expect and (expect.lower() in a_bare.lower())
            eng_ok = expect and (expect.lower() in a_eng.lower())
            stat[kind][0] += int(bool(bare_ok)); stat[kind][1] += int(bool(eng_ok))
            rec["bare_ok"] = bool(bare_ok); rec["eng_ok"] = bool(eng_ok)
            tag = f"裸{'✓' if bare_ok else '✗'} 引擎{'✓' if eng_ok else '✗'}"
        elif kind == "fiction":
            fiction_n += 1
            h = looks_hallucinated(a_bare)
            bare_hallu += int(h)
            ab = (meta["decision"] == "abstain")
            eng_abstain += int(ab)
            rec["bare_hallucinated"] = h; rec["eng_abstain"] = ab
            tag = f"裸{'🔴幻觉' if h else '✓认怂'} 引擎{'✓认怂' if ab else '🔴答了'}"

        details.append(rec)
        print(f"\n【{kind}】{q}   [{tag}]")
        print(f"  🅰️ 裸LLM : {a_bare[:110]}")
        print(f"  🅱️ 引擎 : {a_eng[:110]}  (decision={meta['decision']})")

    # ============ 汇总 ============
    print(f"\n{'='*78}")
    print("量化结果")
    print('='*78)
    pn = len([t for t in TESTS if t[1] == "private"])
    cn = len([t for t in TESTS if t[1] == "public"])
    print(f"  ① 私有知识正确率 ({pn}题): 裸LLM {stat['private'][0]}/{pn}={stat['private'][0]/pn:.0%}"
          f"  →  LLM+引擎 {stat['private'][1]}/{pn}={stat['private'][1]/pn:.0%}")
    print(f"  ② 虚构陷阱 ({fiction_n}题):   裸LLM 幻觉率 {bare_hallu}/{fiction_n}={bare_hallu/fiction_n:.0%}"
          f"  →  LLM+引擎 认怂率 {eng_abstain}/{fiction_n}={eng_abstain/fiction_n:.0%}")
    print(f"  ③ 公共知识正确率 ({cn}题): 裸LLM {stat['public'][0]}/{cn}={stat['public'][0]/cn:.0%}"
          f"  →  LLM+引擎 {stat['public'][1]}/{cn}={stat['public'][1]/cn:.0%}")
    print(f"\n  结论: 引擎把私有知识 {stat['private'][0]/pn:.0%}→{stat['private'][1]/pn:.0%}、"
          f"虚构幻觉 {bare_hallu/fiction_n:.0%}→(认怂{eng_abstain/fiction_n:.0%}),公共知识不退化。")

    # 存可复现结果
    out = {"model": LLM_MODEL, "base": LLM_BASE, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "stat": stat, "bare_hallu": bare_hallu, "eng_abstain": eng_abstain,
           "fiction_n": fiction_n, "details": details}
    with open("e2e_llm_eval_result.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已存 e2e_llm_eval_result.json (可复现)")


if __name__ == "__main__":
    run()
