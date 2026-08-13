#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可靠记忆引擎 交互演示
================================
直观展示两大能力:
  ① 持续学习:教它新知识→立刻记住(不重训),用同义问法也能答
  ② 抗幻觉:  问没教过的→诚实说"我不确定"(绝不瞎编)
支持两种模式:
  python3 demo.py           # 脚本化对话演示(自动跑一串对话,看效果)
  python3 demo.py -i        # 交互模式(你自己输入)
"""
import sys
from reliable_memory_engine import ReliableMemoryEngine

BANNER = """
╔══════════════════════════════════════════════════════════╗
║           可靠记忆引擎 · 会持续学习 + 绝不幻觉              ║
╚══════════════════════════════════════════════════════════╝
"""

def teach(eng, q, a, paras=None):
    ok, r = eng.learn(q, a, source="user", paraphrases=paras)
    print(f"  📚 教它: 「{q}」= {a}   [{r}]")

def ask(eng, q):
    ans, meta = eng.recall(q)
    if meta["decision"] == "answer":
        print(f"  ❓ 问: 「{q}」")
        print(f"     ✅ 答: {ans}   (相似度{meta['sim']:.2f} 可信度{meta['conf']:.2f})")
    else:
        print(f"  ❓ 问: 「{q}」")
        print(f"     🤔 答: {ans}   [诚实认怂·不瞎编]")

def scripted(eng):
    print(BANNER)
    print("【场景1】刚开始,引擎什么都不知道 —— 问它,它诚实认怂\n")
    ask(eng, "小明的生日是哪天")

    print("\n【场景2】现在教它一些知识(不重训,即时学习)\n")
    teach(eng, "小明的生日是哪天", "3月15日", ["小明什么时候生日", "小明的出生日期"])
    teach(eng, "公司WiFi密码是多少", "tanxun2026", ["办公室无线密码", "WiFi怎么连"])
    teach(eng, "项目上线时间", "2026年8月1日", ["项目什么时候发布", "上线日期是"])

    print("\n【场景3】立刻就会了 —— 而且用【没教过的同义问法】也能答\n")
    ask(eng, "小明的生日是哪天")           # 原问法
    ask(eng, "小明哪天过生日")             # 同义改写(没教过这个问法)
    ask(eng, "办公室无线密码是啥")         # 同义改写
    ask(eng, "这个项目啥时候发布")         # 同义改写

    print("\n【场景4】问它没教过的 —— 绝不瞎编,诚实认怂(抗幻觉)\n")
    ask(eng, "小红的生日是哪天")           # 教了小明,没教小红→认怂
    ask(eng, "隔壁公司的WiFi密码")         # 没教→认怂
    ask(eng, "火星基地的开放时间")         # 虚构→认怂

    print("\n【场景5】纠错:知识变了,直接教新的(持续更新,不重训)\n")
    teach(eng, "项目上线时间", "2026年9月1日(推迟了)", ["项目什么时候发布"])
    ask(eng, "项目什么时候上线")

    print("\n【场景6】睡眠固化:高频知识固化进长期记忆\n")
    for _ in range(2): eng.recall("小明的生日是哪天")
    eng.feedback("小明的生日是哪天", accepted=True)
    n = eng.sleep(min_hits=1)
    print(f"  💤 睡眠固化 {n} 条高价值知识进长期记忆")
    print(f"  📊 引擎状态: {eng.stats()}")

    print("\n" + "="*60)
    print("演示总结:")
    print("  ① 持续学习: 教了就会,不重训,同义问法也能答 ✅")
    print("  ② 抗幻觉:   没教过的一律诚实认怂,绝不瞎编 ✅")
    print("  ③ 可更新:   知识变了直接教新的,持续更新 ✅")
    print("  ④ 睡眠固化: 高价值知识固化进长期记忆 ✅")
    print("="*60)

def interactive(eng):
    print(BANNER)
    print("交互模式。命令:")
    print("  学 问题=答案      教它新知识")
    print("  <直接输入问题>    问它(会答或诚实认怂)")
    print("  状态 / 退出\n")
    while True:
        try:
            s = input(">>> ").strip()
        except EOFError:
            break
        if not s: continue
        if s in ("退出", "quit", "exit"): break
        if s == "状态":
            print(f"  {eng.stats()}"); continue
        if s.startswith("学 ") and "=" in s:
            q, a = s[2:].split("=", 1)
            teach(eng, q.strip(), a.strip())
        else:
            ask(eng, s)

if __name__ == "__main__":
    print("初始化引擎(bge-small-zh)...")
    eng = ReliableMemoryEngine()
    if "-i" in sys.argv:
        interactive(eng)
    else:
        scripted(eng)
