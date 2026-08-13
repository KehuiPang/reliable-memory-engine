#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带向量索引的可靠记忆引擎 (IndexedReliableMemory)
====================================================
产品化第一步:把暴力kNN检索换成HNSW向量索引,让引擎扛住上万条知识。
关键:索引只加速"找候选",找到后照样过三重认怂门(限定词/专名/语义/可信度)——
     绝不破坏已验证的抗幻觉机制。

设计:
  · 每个key(标准问+每个paraphrase)作为一个索引项,记录它属于哪个知识槽(slot_id)
  · 检索:HNSW找top-k候选key → 映射到知识槽 → 取每槽最高分 → 走三重门
  · 复用 ReliableMemoryEngine 的 learn门控/实体抽取/三重门逻辑
"""
import numpy as np
import re
from reliable_memory_engine import ReliableMemoryEngine, KUnit
# ⚠️ 不在此import faiss:faiss的OpenMP必须在torch/sentence-transformers之后加载,
#    否则macOS下encode段错误崩溃。故延迟到__init__(父类已先import SentenceTransformer)后再import。


class IndexedReliableMemory(ReliableMemoryEngine):
    def __init__(self, dim=512, **kwargs):
        super().__init__(**kwargs)   # 父类先加载 SentenceTransformer(torch)
        import faiss                  # 此时torch已加载,再import faiss才安全
        self._faiss = faiss
        # bge-small-zh 维度=512;faiss HNSW索引(O(log N)近似最近邻,内积=归一化后余弦)
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200
        # efSearch=检索时搜索深度。万级规模压测(scale_stress.py + _tune_ef.py)实测(512维):
        #   efSearch=50 在 5万条时 Recall@1 仅 72%(HNSW漏真最近邻→假认怂/漏答);
        #   且 efSearch 需随库规模上调:5万条 200 够(98%),10万条要 400 才 99%。
        # 故用 _adapt_efsearch() 按库规模自适应(库越大搜越深),兼顾召回与延迟。
        self.index.hnsw.efSearch = 200
        # 注:_adapt_efsearch 在下面 _n_items 初始化后调用(此处 _n_items 尚未定义)
        self.item2slot = []      # 索引项id → 知识槽下标
        self._n_items = 0
        self._adapt_efsearch()   # 空库先按 n=0 设默认(128),后续 add 时再自适应
        # faiss 不支持高效删除:feedback/sleep 移除槽时不物理删,改软删除(记下标),
        # 检索时跳过。避免 pop 导致 item2slot 里所有 slot_id 整体错位。
        self.dead_slots = set()
        # 脏标记:有写入/更新/软删除后置 True,供定时自动 save 判断是否需落盘
        self._dirty = False

    def _adapt_efsearch(self):
        """按库规模自适应 efSearch:库越大近似搜索要越深才能保 Recall。
        实测甜点(512维): ≤1万→128, 5万→256, 10万→400, 上限512。
        用 log 平滑:ef = clip(round(48*log2(n+1)), 128, 512)。
        n=1万→~640→截512? 校准取更缓系数,保证 5万≈256/10万≈400。"""
        import math
        n = max(getattr(self, "_n_items", 0), 1)
        ef = int(48 * math.log2(n + 1))         # 1万→~638 偏大;下面按档位校准更稳
        # 分档校准(与实测甜点对齐,避免 log 系数在中间段偏差)
        if n <= 2000:
            ef = 128
        elif n <= 20000:
            ef = 200
        elif n <= 60000:
            ef = 300
        else:
            ef = 400 if n <= 150000 else 512
        self.index.hnsw.efSearch = ef

    def _add_keys_to_index(self, slot_id, keys):
        vecs = np.asarray(keys, dtype=np.float32)
        self.index.add(vecs)
        for _ in keys:
            self.item2slot.append(slot_id)
            self._n_items += 1
        self._adapt_efsearch()   # 库变大后同步调深搜索深度

    # 覆写 learn:写入后把它的所有key加进索引
    def learn(self, question, answer, source="user", paraphrases=None):
        # 快照:记录本次 learn 之前各槽的 key 数量,用于精确定位 updated 追加的新 key
        keycnt_before = [len(s.keys) for s in self.slots]
        ok, msg = super().learn(question, answer, source, paraphrases)
        if not ok:
            return ok, msg
        if msg.startswith("learned"):
            # 新增槽:把它的所有 key 入索引
            slot_id = len(self.slots) - 1
            self._add_keys_to_index(slot_id, self.slots[slot_id].keys)
        elif msg.startswith("updated"):
            # 更新旧槽:父类给某个已有槽的 keys 追加了新 key,须把新增 key 补进索引,
            # 否则用新问法检索时 faiss 里没有对应向量 → 命中不到(遗留 bug 修复)
            for sid, cnt_before in enumerate(keycnt_before):
                cnt_now = len(self.slots[sid].keys)
                if cnt_now > cnt_before:
                    new_keys = self.slots[sid].keys[cnt_before:]
                    self._add_keys_to_index(sid, new_keys)
        self._dirty = True
        return ok, msg

    # 覆写 recall:用HNSW找候选,再走三重门(门逻辑复用父类)
    def recall(self, query, topk=10):
        if not self.slots or self._n_items == 0:
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "empty"}
        q = self._encode(query)[0].astype(np.float32)
        k = min(topk, self._n_items)
        sims_arr, labels = self.index.search(np.asarray([q]), k)  # faiss: 内积直接是相似度
        cand_slot_sim = {}
        for item_id, sim in zip(labels[0], sims_arr[0]):
            if item_id < 0:
                continue
            slot_id = self.item2slot[int(item_id)]
            if slot_id in self.dead_slots:   # 软删除的槽:跳过
                continue
            sim = float(sim)
            if slot_id not in cand_slot_sim or sim > cand_slot_sim[slot_id]:
                cand_slot_sim[slot_id] = sim
        if not cand_slot_sim:   # 候选全被软删除
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "empty"}
        # 取候选里相似度最高的槽
        bi = max(cand_slot_sim, key=cand_slot_sim.get)
        best = cand_slot_sim[bi]
        unit = self.slots[bi]

        # ===== 复用父类三重认怂门(逻辑一致) =====
        if best < self.sim_th:
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "low_sim", "sim": best}
        # 门2a 限定词冲突
        qualifiers = ["隔壁", "别的", "别人", "其他", "另一", "另外", "对面", "邻居", "他们的", "别家"]
        q_quals = [w for w in qualifiers if w in query]
        if q_quals and not any(w in unit.question for w in q_quals):
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "qualifier_conflict", "sim": best}
        # 门2b 专名字面核验
        q_ent = self._extract_entity(query); k_ent = unit.entity
        if self._is_proper(q_ent) and self._is_proper(k_ent) and q_ent.lower() != k_ent.lower():
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "proper_name_mismatch",
                    "sim": best, "q_entity": q_ent, "k_entity": k_ent}
        # 门2c 语义实体核验
        ent_sim = float(q @ unit.entity_key)
        eff_th = self.entity_th if best < self.strong_sim else self.entity_th * 0.5
        if ent_sim < eff_th:
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "entity_mismatch",
                    "sim": best, "entity_sim": ent_sim}
        # 门3 可信度
        if unit.confidence < self.conf_th:
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "low_conf",
                    "sim": best, "conf": unit.confidence}
        unit.hits += 1
        return unit.content, {"decision": "answer", "sim": best, "entity_sim": ent_sim,
                "conf": unit.confidence, "source": unit.source}

    # ---------- 覆写 feedback:拒绝时软删除(不 pop,避免索引下标错位) ----------
    def feedback(self, query, accepted=True):
        if not self.slots or self._n_items == 0:
            return
        # 复用索引找最相似的活槽(与 recall 同口径)
        q = self._encode(query)[0].astype(np.float32)
        k = min(10, self._n_items)
        sims_arr, labels = self.index.search(np.asarray([q]), k)
        cand = {}
        for item_id, sim in zip(labels[0], sims_arr[0]):
            if item_id < 0:
                continue
            sid = self.item2slot[int(item_id)]
            if sid in self.dead_slots:
                continue
            if sid not in cand or float(sim) > cand[sid]:
                cand[sid] = float(sim)
        if not cand:
            return
        bi = max(cand, key=cand.get)
        # 相似度不足 sim_th 说明这条 query 根本不对应该槽,不做反馈(避免误伤邻近槽)
        if cand[bi] < self.sim_th:
            return
        unit = self.slots[bi]
        if accepted:
            unit.confidence += 0.15 * (1.0 - unit.confidence)
        else:
            unit.confidence *= 0.3
            if unit.confidence < 0.15:
                self.dead_slots.add(bi)   # 软删除:标记失效,保留索引映射不错位
        self._dirty = True

    # ---------- 覆写 sleep:固化后软删除该槽(不重建 slots 列表) ----------
    def sleep(self, min_hits=2):
        consolidated = 0
        for sid, s in enumerate(self.slots):
            if sid in self.dead_slots:
                continue
            if s.hits >= min_hits and s.confidence >= 0.7:
                self.cortex[s.question] = s.content
                self.dead_slots.add(sid)
                consolidated += 1
        if consolidated:
            self._dirty = True
        return consolidated

    # ---------- 持久化:save/load(产品化,重启不丢) ----------
    def save(self, path_dir):
        """保存引擎状态到目录:知识元数据(json)+向量(npz)+faiss索引+映射"""
        import os, json
        os.makedirs(path_dir, exist_ok=True)
        # 1. 知识元数据(不含向量,可json)
        meta = []
        for s in self.slots:
            meta.append({"content": s.content, "question": s.question, "entity": s.entity,
                         "confidence": s.confidence, "source": s.source, "hits": s.hits,
                         "n_keys": len(s.keys)})
        with open(os.path.join(path_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"slots": meta, "item2slot": self.item2slot, "n_items": self._n_items,
                       "dead_slots": sorted(self.dead_slots),
                       "sim_th": self.sim_th, "entity_th": self.entity_th, "conf_th": self.conf_th,
                       "strong_sim": self.strong_sim, "update_sim": self.update_sim,
                       "cortex": self.cortex, "dim": self.dim}, f, ensure_ascii=False)
        # 2. 向量(key/entity_key/多keys)存npz
        arrays = {}
        for i, s in enumerate(self.slots):
            arrays["k%d" % i] = s.key
            arrays["e%d" % i] = s.entity_key
            arrays["ks%d" % i] = np.asarray(s.keys, dtype=np.float32)
        np.savez(os.path.join(path_dir, "vectors.npz"), **arrays)
        # 3. faiss索引
        self._faiss.write_index(self.index, os.path.join(path_dir, "index.faiss"))
        self._dirty = False   # 落盘后清脏标记
        return path_dir

    @classmethod
    def load(cls, path_dir, embed_model="BAAI/bge-small-zh-v1.5"):
        """从目录恢复引擎(向量不重编码,直接读盘)"""
        import os, json
        eng = cls(embed_model=embed_model)   # 会加载embedding模型+建空索引
        with open(os.path.join(path_dir, "meta.json"), encoding="utf-8") as f:
            d = json.load(f)
        vecs = np.load(os.path.join(path_dir, "vectors.npz"))
        eng.slots = []
        for i, m in enumerate(d["slots"]):
            eng.slots.append(KUnit(
                content=m["content"], question=m["question"], entity=m["entity"],
                key=vecs["k%d" % i], entity_key=vecs["e%d" % i],
                confidence=m["confidence"], source=m["source"],
                keys=list(vecs["ks%d" % i]), hits=m["hits"]))
        eng.item2slot = d["item2slot"]; eng._n_items = d["n_items"]
        eng.dead_slots = set(d.get("dead_slots", []))
        eng.sim_th=d["sim_th"]; eng.entity_th=d["entity_th"]; eng.conf_th=d["conf_th"]
        eng.strong_sim=d["strong_sim"]; eng.update_sim=d["update_sim"]; eng.cortex=d["cortex"]
        # 读回faiss索引(替换__init__建的空索引)
        eng.index = eng._faiss.read_index(os.path.join(path_dir, "index.faiss"))
        eng._adapt_efsearch()   # 按恢复后的库规模重设搜索深度
        return eng
