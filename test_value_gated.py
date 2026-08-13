#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ValueGatedMemory 整合验证:A6-A8 价值门控接进引擎 learn 链路
验证:
  1) 可信度门:低可信来源(source=web/unknown)被拦
  2) 价值门:高价值放行、低价值缓存
  3) 探索配额:低价值但够新奇的知识有机会被探索学入
  4) 命中回流:record_hit 后 V̂ 权重更新(在线慢学)
  5) 【关键】recall 三重认怂门零退化:抗幻觉机制不受价值门控影响
在 engine 目录跑: python3 test_value_gated.py
"""
from value_gated_engine import ValueGatedMemory
import numpy as np

def line(t): print("\n" + "=" * 8 + " " + t + " " + "=" * 8)

eng = ValueGatedMemory(value_gate=True, explore_eps=0.0, seed=0)  # 先关探索,测确定性行为

line("测试1: 可信度门拦低可信来源")
ok, msg = eng.learn("某匿名爆料说X公司要倒闭", "X公司要倒闭", source="web")
print("  web来源 →", ok, msg)
assert not ok and "low_cred" in msg

line("测试2: 高价值(用户权威)放行 + 命中")
ok, msg = eng.learn("figcheck什么时候上线", "figcheck于2023年上线", source="user", sem_hint=0.6)
print("  user高价值 →", ok, msg)
assert ok
ans, info = eng.recall("figcheck啥时候上线的")
print("  recall →", ans, "|", info["decision"])
assert info["decision"] == "answer" and "2023" in ans

line("测试3: 低价值→缓存观察(不直接写)")
ok, msg = eng.learn("某个几乎用不到的冷门参数", "值是42", source="user", sem_hint=-0.9)
print("  低价值 →", ok, msg)
assert not ok and "cached" in msg
print("  缓存区:", len(eng.cache), "条")

line("测试4: 探索配额生效(开ε=1.0,低价值但新奇也给机会)")
eng2 = ValueGatedMemory(value_gate=True, explore_eps=1.0, seed=1)
ok, msg = eng2.learn("一个全新领域的罕见知识", "罕见答案", source="user", sem_hint=-0.5)
print("  ε=1.0低价值新奇 →", ok, msg)
assert ok and "explore" in msg, "探索配额应给新奇知识机会"

line("测试5: 命中回流更新 V̂(在线慢学)")
w_before = eng.vhead.w.copy()
for _ in range(5):
    eng.record_hit("figcheck啥时候上线的", success=True, adopted=True)
w_after = eng.vhead.w
moved = float(np.linalg.norm(w_after - w_before))
print("  V̂权重移动量 =", round(moved, 5), "(应>0,证明在线学习)")
assert moved > 0

line("测试6: 【关键】抗幻觉三重门零退化")
# 价值门控不应影响 recall 的认怂能力:虚构问题仍认怂
ans, info = eng.recall("瓦坎达的首都是哪")
print("  虚构问题 →", ans, "|", info["decision"])
assert info["decision"] == "abstain", "抗幻觉退化!价值门控破坏了recall三重门"
# 限定词陷阱仍认怂
eng.learn("公司WiFi密码是多少", "密码是tanxun2023", source="user", sem_hint=0.5)
ans, info = eng.recall("隔壁公司WiFi密码是多少")
print("  限定词陷阱 →", ans, "|", info["decision"])
assert info["decision"] == "abstain", "限定词门退化!"

print("\n门控统计:", eng.gate_stats())
print("\n" + "=" * 40)
print("🎉 价值门控整合验证全部通过:A6-A8已接进引擎,抗幻觉零退化")
