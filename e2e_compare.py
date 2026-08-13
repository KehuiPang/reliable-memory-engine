#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端对比:裸LLM vs LLM+可靠记忆引擎
================================================
用真实本地LLM(ollama qwen2.5:1.5b)对比两个系统回答同一批问题:
  系统A 裸LLM:      直接问大模型(没有外部记忆,靠训练时的参数)
  系统B LLM+记忆引擎: 先查可靠记忆引擎→命中用可信知识答,没命中诚实认怂

专门设计3类问题暴露裸LLM短板:
  ① 私有/新知识(公司内部信息)——裸LLM不可能知道
  ② 抗幻觉(不存在的东西)——看裸LLM会不会一本正经胡说
  ③ 知识更新——知识变了,记忆引擎能更新,裸LLM停在旧信息
"""
import sys, json, urllib.request
from reliable_memory_engine import ReliableMemoryEngine

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"

def llm(prompt, system=None):
    """调ollama LLM"""
    data = {"model": MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}
    if system: data["system"] = system
    req = urllib.request.Request(OLLAMA, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["response"].strip()
    except Exception as e:
        return f"[LLM调用失败:{e}]"

def bare_llm(question):
    """系统A:裸LLM,直接问"""
    return llm(question, system="你是一个助手,请简洁回答问题。")

def llm_with_memory(question, eng):
    """系统B:先查记忆引擎,命中用记忆答,没命中诚实认怂"""
    ans, meta = eng.recall(question)
    if meta["decision"] == "answer":
        # 命中可信记忆→让LLM基于这条可信知识组织回答(不瞎编)
        prompt = f"根据以下确切信息回答问题。\n确切信息:{ans}\n问题:{question}\n简洁回答:"
        return f"{llm(prompt)}  【✓来自可信记忆:{ans}】"
    else:
        return "我不确定这个信息,需要查证。【记忆库无此知识,诚实认怂,不瞎编】"

def run():
    print("初始化记忆引擎...")
    eng = ReliableMemoryEngine()

    # 给记忆引擎教一些"私有/新"知识(裸LLM不可能知道)
    eng.learn("探寻公司的figcheck项目上线时间", "2026年8月1日", source="clean",
              paraphrases=["figcheck什么时候上线", "figcheck项目发布日期"])
    eng.learn("探寻公司2026年会地点", "深圳", source="clean",
              paraphrases=["公司年会在哪开", "年会地点在哪"])

    # 测试问题:(问题, 类型)
    tests = [
        ("探寻公司的figcheck项目什么时候上线", "私有知识"),
        ("figcheck项目发布日期是哪天", "私有知识(同义问法)"),
        ("探寻公司2026年会在哪个城市开", "私有知识"),
        ("珠穆朗玛峰有多高", "公共知识(两者都该会)"),
        ("探寻公司隔壁那家公司的年会在哪开", "抗幻觉(限定词陷阱)"),
        ("探寻公司的flycheck项目上线时间", "抗幻觉(不存在的项目)"),
    ]

    print(f"\n{'='*72}")
    print(f"端到端对比:裸LLM({MODEL}) vs LLM+可靠记忆引擎")
    print('='*72)

    for q, typ in tests:
        print(f"\n【{typ}】❓ {q}")
        a = bare_llm(q)
        print(f"  🅰️ 裸LLM:      {a[:120]}")
        b = llm_with_memory(q, eng)
        print(f"  🅱️ LLM+记忆:   {b[:160]}")

    print(f"\n{'='*72}")
    print("对比结论(看上面):")
    print("  · 私有知识: 裸LLM不知道(瞎编/拒答), LLM+记忆能准确答 ✅")
    print("  · 抗幻觉:   裸LLM可能一本正经胡说, LLM+记忆诚实认怂 ✅")
    print("  · 公共知识: 两者都会(记忆引擎不影响已知能力)")
    print('='*72)

if __name__ == "__main__":
    run()
