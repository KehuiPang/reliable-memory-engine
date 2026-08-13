#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可靠记忆引擎 (ReliableMemoryEngine)
====================================
整合 A1-A3 + 规模化 + 实体核验 + demo修复 全部已验证机制的通用引擎。
对应 08文档"落地轨:通用记忆引擎"。设计哲学:"可以漏答,绝不瞎编"。
不绑任何业务/大模型:embedding可插拔,知识是纯文本KV。
"""
import numpy as np
import re
from dataclasses import dataclass, field


@dataclass
class KUnit:
    """可信知识单元(13文档):带可信度/来源/实体/使用记录"""
    content: str
    question: str
    entity: str
    key: np.ndarray
    entity_key: np.ndarray
    confidence: float
    source: str
    keys: list = field(default_factory=list)   # 多key:标准问+paraphrase(20文档)
    hits: int = 0


class ReliableMemoryEngine:
    def __init__(self, embed_model="BAAI/bge-small-zh-v1.5",
                 sim_th=0.45, entity_th=0.55, conf_th=0.5, strong_sim=0.90):
        from sentence_transformers import SentenceTransformer
        self.enc = SentenceTransformer(embed_model)
        self.slots = []
        self.sim_th = sim_th          # 相似度门
        self.entity_th = entity_th    # 实体核验门(语义级)
        self.conf_th = conf_th        # 可信度门
        self.strong_sim = strong_sim  # 强命中阈值:超过则放宽语义实体核验
        self.update_sim = 0.85        # 更新旧槽阈值
        self.cortex = {}              # 睡眠固化的长期记忆

    # ---------- 工具 ----------
    def _encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        v = self.enc.encode(texts, normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def _extract_entity(self, question):
        """抽关键实体:优先英文/数字专名(figcheck),否则句首中文名词片段"""
        proper = re.findall(r'[A-Za-z][A-Za-z0-9]{2,}', question)
        if proper:
            return proper[0]
        m = re.match(r'^([\u4e00-\u9fa5]+?)(的|是|在|最|主要|被|能|叫|号称)', question)
        return m.group(1) if m else question[:4]

    def _is_proper(self, s):
        return re.fullmatch(r'[A-Za-z0-9]+', s) is not None

    # ---------- 写链路:价值门控 + 更新旧槽 ----------
    def learn(self, question, answer, source="user", paraphrases=None):
        key = self._encode(question)[0]
        entity = self._extract_entity(question)
        entity_key = self._encode(entity)[0]

        source_prior = 0.9 if source in ("clean", "authoritative", "verified") else \
                       0.75 if source == "user" else 0.2
        stability = 1.0
        if paraphrases:
            pk = self._encode(paraphrases)
            stability = float(np.clip((pk @ key).mean(), 0, 1))
        confidence = source_prior * (0.5 + 0.5 * stability)

        if confidence < self.conf_th:
            return False, "gate_reject(conf=%.2f)" % confidence

        all_keys = [key]
        if paraphrases:
            all_keys += [pk_ for pk_ in self._encode(paraphrases)]

        # 更新旧槽(02文档):已有高相似同实体旧知识→覆盖答案而非追加
        if self.slots:
            sims = np.array([max(float(k @ key) for k in (s.keys if s.keys else [s.key]))
                             for s in self.slots])
            bi = int(np.argmax(sims))
            old = self.slots[bi]
            ent_sim = float(entity_key @ old.entity_key)
            if sims[bi] >= self.update_sim and ent_sim >= self.entity_th:
                old.content = answer
                old.confidence = max(old.confidence, confidence)
                for nk in all_keys:
                    old.keys.append(nk)
                return True, "updated(conf=%.2f)" % old.confidence

        self.slots.append(KUnit(answer, question, entity, key, entity_key,
                                confidence, source, keys=all_keys))
        return True, "learned(conf=%.2f)" % confidence

    # ---------- 读链路:多重认怂门 ----------
    def recall(self, query):
        if not self.slots:
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "empty"}
        q = self._encode(query)[0]
        sims = np.array([max(float(k @ q) for k in (s.keys if s.keys else [s.key]))
                         for s in self.slots])
        bi = int(np.argmax(sims))
        best = float(sims[bi]); unit = self.slots[bi]

        # 门1 相似度
        if best < self.sim_th:
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "low_sim", "sim": best}

        # 门2a 限定词冲突(隔壁公司的WiFi ≠ 公司WiFi)
        qualifiers = ["隔壁", "别的", "别人", "其他", "另一", "另外", "对面", "邻居", "他们的", "别家"]
        q_quals = [w for w in qualifiers if w in query]
        if q_quals and not any(w in unit.question for w in q_quals):
            return "[我不确定/需要查证]", {"decision": "abstain", "reason": "qualifier_conflict",
                    "sim": best, "qualifier": q_quals[0]}

        # 门2b 专名字面核验(figcheck ≠ flycheck)
        q_ent = self._extract_entity(query)
        k_ent = unit.entity
        if self._is_proper(q_ent) and self._is_proper(k_ent):
            if q_ent.lower() != k_ent.lower():
                return "[我不确定/需要查证]", {"decision": "abstain", "reason": "proper_name_mismatch",
                        "sim": best, "q_entity": q_ent, "k_entity": k_ent}

        # 门2c 语义实体核验(强命中放宽,避免同义指代误伤)
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

    # ---------- 价值度反馈 ----------
    def feedback(self, query, accepted=True):
        if not self.slots:
            return
        q = self._encode(query)[0]
        sims = np.array([max(float(k @ q) for k in (s.keys if s.keys else [s.key]))
                         for s in self.slots])
        bi = int(np.argmax(sims)); unit = self.slots[bi]
        if accepted:
            unit.confidence += 0.15 * (1.0 - unit.confidence)
        else:
            unit.confidence *= 0.3
            if unit.confidence < 0.15:
                self.slots.pop(bi)

    # ---------- 睡眠固化 ----------
    def sleep(self, min_hits=2):
        consolidated = 0
        remain = []
        for s in self.slots:
            if s.hits >= min_hits and s.confidence >= 0.7:
                self.cortex[s.question] = s.content
                consolidated += 1
            else:
                remain.append(s)
        self.slots = remain
        return consolidated

    def stats(self):
        return {"记忆库条数": len(self.slots), "皮层固化条数": len(self.cortex),
                "sim阈值": round(self.sim_th, 2), "实体阈值": round(self.entity_th, 2)}
