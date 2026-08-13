#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规模化性能对比:暴力kNN引擎 vs faiss索引引擎
证明:①索引让检索又快(O(logN)) ②又准(三重门结果一致,抗幻觉不破坏)
造上万条知识,对比查询速度和命中率/诚实率。
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'data'))
from reliable_memory_engine import ReliableMemoryEngine
from indexed_engine import IndexedReliableMemory
from gen_knowledge_v2 import gen_knowledge_v2, gen_unknown_v2

def gen_bulk(n):
    """造n条有区分度的合成知识:用V2真实知识+大量唯一编号知识凑规模(但每条实体唯一区分)"""
    base = gen_knowledge_v2()  # 63条真实
    items = list(base)
    i = 0
    cities=["北京","上海","广州","深圳","杭州","成都","武汉","西安","南京","重庆"]
    while len(items) < n:
        c = cities[i % len(cities)]
        # 每条用唯一编号做实体,保证区分度(模拟真实系统里大量唯一实体)
        items.append({"q": f"编号{i}号档案的登记城市是哪", "a": c, "src":"clean",
                      "paraphrases":[f"档案{i}号在哪个城市登记", f"{i}号档案登记地"]})
        i += 1
    return items[:n]

def build(engine, items):
    t0=time.time()
    for i, it in enumerate(items):
        engine.learn(it["q"], it["a"], source="clean", paraphrases=it.get("paraphrases"))
        if (i+1) % 500 == 0:
            print(f"    ...已建 {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    return time.time()-t0

def eval_engine(engine, test_items, unknowns):
    # 命中率(用paraphrase查)
    t0=time.time()
    hit=tot=0
    for it in test_items:
        for pq in it["paraphrases"]:
            a,m=engine.recall(pq)
            if m["decision"]=="answer" and a==it["a"]: hit+=1
            tot+=1
    query_time=time.time()-t0
    ab=sum(1 for u in unknowns if engine.recall(u)[1]["decision"]=="abstain")
    return hit/tot, ab/len(unknowns), query_time, tot

def run():
    N = 1500   # 规模(CPU编码慢,1500足够对比速度;真实系统可到百万级)
    print(f"造 {N} 条合成知识...")
    items = gen_bulk(N)
    unk = gen_unknown_v2()
    test = items[:63]  # 用前63条真实知识做命中测试(有自然paraphrase)

    print(f"\n{'='*66}")
    print(f"规模化性能对比 (知识库 {N} 条)")
    print('='*66)

    # 暴力kNN引擎
    print("\n[暴力kNN引擎] 建库中...")
    e1 = ReliableMemoryEngine()
    bt1 = build(e1, items)
    h1,hon1,qt1,nq = eval_engine(e1, test, unk)
    print(f"  建库{bt1:.1f}s | {nq}次查询{qt1:.2f}s ({1000*qt1/nq:.1f}ms/次) | 命中{h1:.0%} 诚实{hon1:.0%}")

    # faiss索引引擎
    print("\n[faiss索引引擎] 建库中...")
    e2 = IndexedReliableMemory()
    bt2 = build(e2, items)
    h2,hon2,qt2,_ = eval_engine(e2, test, unk)
    print(f"  建库{bt2:.1f}s | {nq}次查询{qt2:.2f}s ({1000*qt2/nq:.1f}ms/次) | 命中{h2:.0%} 诚实{hon2:.0%}")

    print(f"\n{'='*66}")
    print("结论")
    print('='*66)
    speedup = qt1/qt2 if qt2>0 else 0
    print(f"  检索速度: 暴力{1000*qt1/nq:.1f}ms/次 → 索引{1000*qt2/nq:.1f}ms/次  加速{speedup:.1f}x")
    print(f"  命中率:   暴力{h1:.0%} vs 索引{h2:.0%}  ({'一致' if abs(h1-h2)<=0.05 else '有差异'})")
    print(f"  诚实率:   暴力{hon1:.0%} vs 索引{hon2:.0%}  ({'一致' if abs(hon1-hon2)<=0.05 else '有差异'})")
    if speedup>1.3 and abs(h1-h2)<=0.05 and abs(hon1-hon2)<=0.05:
        print(f"\n  ✅ 索引又快又准:检索加速{speedup:.1f}x,命中率/诚实率与暴力kNN一致(三重门未破坏)")

if __name__ == "__main__":
    run()
