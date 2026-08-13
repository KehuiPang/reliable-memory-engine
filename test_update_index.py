#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回归测试:验证 indexed_engine 三个遗留 bug 的修复
  1) updated 追加的新 key 能入 faiss 索引 → 用新问法可命中
  2) feedback 拒绝软删除后不再命中,且索引下标不错位(其他槽仍正常)
  3) save/load 保留 dead_slots(软删除状态)
在 engine 目录下跑: python3 test_update_index.py
"""
import os, shutil, sys
from indexed_engine import IndexedReliableMemory

def line(t): print("\n" + "=" * 8 + " " + t + " " + "=" * 8)

eng = IndexedReliableMemory()

line("准备:写入两条知识")
eng.learn("figcheck 什么时候上线的", "figcheck 于 2023 年上线", source="user")
eng.learn("公司年会在哪开", "年会在三亚开", source="user")
print("slots=%d items=%d" % (len(eng.slots), eng._n_items))

line("测试1: updated 追加新 key 入索引")
# 用高相似同实体的问法更新 figcheck 那条,并带新 paraphrase
ok, msg = eng.learn("figcheck 是什么时候上线的呀",
                    "figcheck 于 2023 年 6 月正式上线",
                    source="user",
                    paraphrases=["figcheck 上线时间", "figcheck 何时发布"])
print("learn 返回:", ok, msg, "| items 现在=%d" % eng._n_items)
assert msg.startswith("updated"), "应触发 updated 分支"
# 用新 paraphrase 的问法检索,修复前会命中不到
ans, info = eng.recall("figcheck 何时发布")
print("用新问法'figcheck 何时发布'检索 →", ans, "|", info.get("decision"))
assert info["decision"] == "answer" and "2023" in ans, "新key未入索引→命中失败(bug未修)"
print("✅ 测试1通过:updated 新 key 已入索引,新问法能命中")

line("测试2: feedback 软删除不错位")
# 反复拒绝年会那条,把置信度打到 <0.15 触发软删除
for _ in range(6):
    eng.feedback("公司年会在哪开", accepted=False)
ans2, info2 = eng.recall("公司年会在哪开")
print("软删除后检索年会 →", ans2, "|", info2["decision"], "| dead_slots=", eng.dead_slots)
assert info2["decision"] == "abstain", "被软删除的槽不应再命中"
# 关键:软删除年会后,figcheck 那条仍能正常命中(证明索引下标没错位)
ans3, info3 = eng.recall("figcheck 什么时候上线的")
print("软删除后检索 figcheck →", ans3, "|", info3["decision"])
assert info3["decision"] == "answer" and "2023" in ans3, "软删除导致其他槽索引错位(bug)"
print("✅ 测试2通过:软删除只失效目标槽,其余槽索引不错位")

line("测试3: save/load 保留 dead_slots")
tmp = "./_test_save_tmp"
if os.path.exists(tmp):
    shutil.rmtree(tmp)
eng.save(tmp)
eng2 = IndexedReliableMemory.load(tmp)
print("load 后 dead_slots=", eng2.dead_slots)
assert eng2.dead_slots == eng.dead_slots, "dead_slots 未持久化"
ans4, info4 = eng2.recall("公司年会在哪开")
print("load 后检索被软删的年会 →", info4["decision"])
assert info4["decision"] == "abstain", "load 后软删除状态丢失"
ans5, info5 = eng2.recall("figcheck 何时发布")
print("load 后用新问法检索 figcheck →", ans5, "|", info5["decision"])
assert info5["decision"] == "answer" and "2023" in ans5, "load 后新key索引丢失"
shutil.rmtree(tmp)
print("✅ 测试3通过:save/load 完整保留软删除+新key索引")

print("\n" + "=" * 30)
print("🎉 全部回归测试通过:三个遗留 bug 已修复")
