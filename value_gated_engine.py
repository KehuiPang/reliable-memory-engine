#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价值门控记忆引擎 (ValueGatedMemory) —— A6-A8 研究成果转产品
==============================================================
把实证过的价值门控(A6价值可回归 + A7信用分配 + A8探索/防漂移)接到可靠记忆引擎的 learn 链路。
设计铁律:
  · 只增强"写入前要不要学"的决策(节流阀),【绝不碰 recall 的三重认怂门】——抗幻觉机制原样保留。
  · 价值门控是可选增强:vge = ValueGatedMemory(); 不启用价值门时行为与 IndexedReliableMemory 一致。
  · 五档决策(02文档四)的在线简化:reject(丢弃)/cache(缓存观察)/learn(写)/update(更新)。
  · 价值预测器在线【慢更新】(A8结论:防漂移),命中回报流回时才更新。

对应实证:
  A6 → 价值预测器 V̂(feat) 离线可回归,这里在线增量版
  A7 → 命中回报按注意力/价值分摊为信用,回流训 V̂
  A8 → 探索配额(ε 给纯新奇度) + 慢更新(lr小) + 可信度门兜底(探索不学噪声)
"""
import numpy as np
from indexed_engine import IndexedReliableMemory


class OnlineValueHead:
    """在线价值预测器 V̂(feat)=w·feat+b,慢更新SGD(A8:lr小防漂移)。
       feat = [novelty惊讶度, credibility可信度, sem语义价值代理]。
    """
    def __init__(self, dim=3, lr=0.02, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, dim)
        self.b = 0.0
        self.lr = lr          # A8:慢更新,宁可迟钝也要稳

    def predict(self, feat):
        return float(self.w @ np.asarray(feat, float) + self.b)

    def update(self, feat, target):
        feat = np.asarray(feat, float)
        err = self.predict(feat) - target
        self.w -= self.lr * err * feat
        self.b -= self.lr * err


class ValueGatedMemory(IndexedReliableMemory):
    def __init__(self, dim=512,
                 value_gate=True,      # 是否启用价值门控(关掉=退化为IndexedReliableMemory)
                 value_th=-0.2,        # 价值门:V̂低于此→丢弃或缓存
                 explore_eps=0.3,      # A8探索配额:这个概率下即使V̂低,只要够新奇也给机会
                 cred_th=0.5,          # 可信度门(探索也不破例,A8防噪)
                 value_lr=0.02,        # A8慢更新
                 seed=0, **kwargs):
        super().__init__(dim=dim, **kwargs)
        self.value_gate = value_gate
        self.value_th = value_th
        self.explore_eps = explore_eps
        self.cred_th = cred_th
        self.vhead = OnlineValueHead(lr=value_lr, seed=seed)
        self._rng = np.random.default_rng(seed)
        self.cache = []          # 缓存观察区(02文档:将信将疑,证据不足暂存)
        self.gate_log = []       # 门控决策日志(可审计)

    # ---------- 惊讶度/特征 ----------
    def _novelty(self, question_vec):
        """惊讶度≈检索失配度:到现有记忆最近邻越远越新奇(A6:免费,前向自带思想)。"""
        if not self.slots or self._n_items == 0:
            return 1.0
        q = np.asarray(question_vec, dtype=np.float32)
        k = min(5, self._n_items)
        sims, _ = self.index.search(np.asarray([q]), k)
        best = float(sims[0][0]) if len(sims[0]) else 0.0
        return float(np.clip(1.0 - best, 0, 1))   # 越不相似越新奇

    def _value_feat(self, question_vec, credibility, sem_hint):
        novelty = self._novelty(question_vec)
        return [novelty, credibility, sem_hint], novelty

    # ---------- 价值门控 learn ----------
    def learn(self, question, answer, source="user", paraphrases=None, sem_hint=None):
        """在原引擎门控前,加一道价值门控(五档决策简化版)。
        sem_hint: 可选的该知识"预期价值"提示(A7:语义特征代理携带部分价值信息);
                  缺省用来源先验粗估。
        """
        if not self.value_gate:
            return super().learn(question, answer, source, paraphrases)

        # 可信度(与父类同口径粗估:来源先验)
        credibility = 0.9 if source in ("clean", "authoritative", "verified") else \
                      0.75 if source == "user" else 0.2
        if sem_hint is None:
            sem_hint = credibility - 0.5   # 无提示时中性

        qvec = self._encode(question)[0]
        feat, novelty = self._value_feat(qvec, credibility, sem_hint)
        vhat_raw = self.vhead.predict(feat)
        # A6文档六-2冷启动:V̂ 训练不足时会瞎猜,此时降低 V̂ 话语权、直接采信 sem_hint。
        #   随命中回流训练次数增多(_train_count),逐渐把话语权交还给 V̂。
        tc = getattr(self, "_train_count", 0)
        warm = min(1.0, tc / 20.0)                 # 0(冷)→1(暖,训练≥20次)
        vhat = warm * vhat_raw + (1 - warm) * sem_hint

        # ===== 五档决策(在线简化) =====
        # 可信度门(A8:探索也不破例,绝不学噪声)
        if credibility < self.cred_th:
            self.gate_log.append(dict(q=question, decision="reject", reason="low_cred",
                                      vhat=round(vhat, 3), cred=round(credibility, 3)))
            return False, "gate_reject(low_cred=%.2f)" % credibility

        decision = None
        if vhat >= self.value_th:
            decision = "learn"                       # 价值够→写
        else:
            # 价值不够,但 A8 探索配额:够新奇则给机会(探索),否则缓存观察
            if novelty > 0.6 and self._rng.random() < self.explore_eps:
                decision = "explore"                 # 探索:纯因新奇而学
            else:
                decision = "cache"                   # 缓存观察(证据不足,暂存)

        if decision == "cache":
            self.cache.append(dict(question=question, answer=answer, source=source,
                                   paraphrases=paraphrases, feat=feat))
            self.gate_log.append(dict(q=question, decision="cache",
                                      vhat=round(vhat, 3), novelty=round(novelty, 3)))
            return False, "cached(vhat=%.2f,novelty=%.2f)" % (vhat, novelty)

        # learn / explore → 真正写入(走父类完整链路,含 updated 逻辑 + 索引 + 抗幻觉不受影响)
        ok, msg = super().learn(question, answer, source, paraphrases)
        if ok:
            # 记下该 slot 的价值特征,供命中回报回流更新 V̂
            sid = len(self.slots) - 1 if msg.startswith("learned") else None
            self.gate_log.append(dict(q=question, decision=decision, vhat=round(vhat, 3),
                                      novelty=round(novelty, 3), msg=msg))
            # 把特征挂到 slot 上(用 entity 字段旁挂,不改 KUnit 结构:存在实例属性字典里)
            if not hasattr(self, "_slot_feat"):
                self._slot_feat = {}
            if sid is not None:
                self._slot_feat[sid] = feat
        return ok, "%s|%s" % (decision, msg)

    # ---------- 命中回报回流:在线慢更新 V̂(A7信用 + A8慢更新) ----------
    def record_hit(self, query, success=True, adopted=True):
        """recall 命中并知道下游结果后调用:算回报→回流训 V̂(慢更新)。"""
        if not hasattr(self, "_slot_feat") or not self._slot_feat:
            return
        # 找命中的 slot(与 recall 同口径:索引最相似的活槽)
        q = self._encode(query)[0].astype(np.float32)
        k = min(5, self._n_items)
        sims, labels = self.index.search(np.asarray([q]), k)
        for item_id in labels[0]:
            if item_id < 0:
                continue
            sid = self.item2slot[int(item_id)]
            if sid in self.dead_slots or sid not in self._slot_feat:
                continue
            # 回报(A7:命中+成功+采纳为正,失败为负)
            reward = (0.5 * 1.0 + 0.8 * (1.0 if success else -0.5)
                      + 0.5 * (1.0 if adopted else 0.0)) - 0.3   # -存储成本
            self.vhead.update(self._slot_feat[sid], reward)
            self._train_count = getattr(self, "_train_count", 0) + 1
            break

    def gate_stats(self):
        from collections import Counter
        c = Counter(g["decision"] for g in self.gate_log)
        return {"决策分布": dict(c), "缓存区条数": len(self.cache),
                "记忆库条数": len(self.slots) - len(self.dead_slots)}
