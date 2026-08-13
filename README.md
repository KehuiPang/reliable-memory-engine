# 可靠记忆引擎 (Reliable Memory Engine)

> 给任意 LLM/Agent 挂上「**会持续学习 + 绝不幻觉**」的记忆层。
> 核心理念:把 LLM 从"自信地骗你"变成"知道就答、不知道就诚实认怂"。
> 属于「持续学习/AGI 研究」的落地成品(设计与实证见 知识宫殿/AGI研究/)。

---

## 这是什么

一个**通用、中立、可插拔**的记忆引擎:
- **会持续学习**:教了就会(不重训)、同义问法也能命中、知识能更新覆盖
- **绝不幻觉**:检索不到/没把握时诚实说"我不确定",不瞎编 —— 由**三重认怂门**机制保证,不靠模型自觉
- **越用越准**:命中被采纳→可信度升,被证伪→降(触发遗忘)
- **会睡眠固化**:高价值知识蒸馏进长期记忆
- **可规模化**:faiss 向量索引,可扩到上万/百万条
- **可部署**:持久化(重启不丢)、多租户 HTTP 服务

**指标**(63条合成知识,真实改写问句):命中率 98% / 诚实率 100%(诚实率=该认怂时认怂,不瞎编)。

---

## 快速开始

### 1. 环境
```bash
pip install "numpy<2" torch==2.0.1 "sentence-transformers==2.2.2" faiss-cpu
```
⚠️ **两个环境坑**(macOS 实测):
- `faiss` 必须在 `torch`/`sentence-transformers` **之后** import,否则 OpenMP 冲突段错误(本项目已在引擎内延迟 import 处理)。
- `faiss-cpu` 会把 numpy 升到 2.x,与 torch 2.0.1 冲突,须 `pip install "numpy<2"` 降回。

### 2. 直接用引擎(Python)
```python
from engine.reliable_memory_engine import ReliableMemoryEngine
eng = ReliableMemoryEngine()                       # embedding 可插拔,默认 bge-small-zh
eng.learn("公司项目上线时间", "2026年8月1日", source="clean",
          paraphrases=["项目什么时候发布"])          # 教它
print(eng.recall("项目啥时候上线"))                  # → ('2026年8月1日', {...answer...})
print(eng.recall("隔壁公司项目上线时间"))            # → ('[我不确定/需要查证]', {...abstain...})
```

### 3. 规模化+持久化(引擎升级版)
```python
from engine.indexed_engine import IndexedReliableMemory
eng = IndexedReliableMemory()      # faiss 向量索引,O(logN)
eng.learn(...); eng.save("./mymem")           # 落盘
eng = IndexedReliableMemory.load("./mymem")   # 重启恢复
```

### 4. 起 HTTP 服务(任意应用可挂载)
```bash
python3 engine/memory_server.py 8899
# 教它
curl -X POST :8899/learn  -d '{"tenant":"myapp","question":"...","answer":"..."}'
# 查它(命中就答,没把握就认怂)
curl -X POST :8899/recall -d '{"tenant":"myapp","query":"..."}'
```
支持多租户(tenant 隔离)、重启自动恢复。

---

## 抗幻觉:三重认怂门

`recall` 时任一门不过 → 认怂"我不确定":
1. **相似度门**:记忆里有没有相关的
2. **限定词冲突**:query 有"隔壁/别的"等而命中知识没有 → 指代变了
3. **专名字面核验**:figcheck ≠ flycheck(一字之差的专名不放过)
4. **语义实体核验**:问的实体和命中知识的实体是不是同一个
5. **可信度门**:命中知识本身够不够可信

> 端到端实测:裸 LLM 会把不存在的"flycheck"编成"阿里开源工具带GitHub链接";挂了本引擎后诚实认怂。

---

## 目录结构

```
engine/                         ← 成品引擎(用这个)
  reliable_memory_engine.py       基础引擎(记忆库+三重门+价值度+睡眠固化)
  indexed_engine.py               规模化版(faiss索引 + save/load持久化)
  memory_server.py                多租户 HTTP 服务
  demo.py                         交互演示(python3 demo.py / -i)
  e2e_compare.py                  裸LLM vs LLM+记忆 端到端对比(需ollama)
  smoke_test.py / scale_perf.py   验收/性能测试

a1_reliable_memory/ ~ a5_entity/  ← 研发过程的分阶段验证(A1记忆库→A2真实鲁棒→
pc_probe/                            A3睡眠固化→规模化→实体核验;含预测编码探针负结果)
```

---

## 设计与实证文档

完整的"为什么这么设计 + 一步步怎么验证出来的",见知识宫殿:
`知识宫殿/AGI研究/`(30篇,从"大模型为什么不会持续学习"到本引擎的完整历程)
- `README_全景导航图.md` — 先读这篇看全貌
- `13_可靠记忆层原理设计.md` — 原理(为什么是这套设计)
- `26_实证阶段总收官报告.md` — 实证阶段全貌
- `30_最终总报告.md` — 整个项目的终极总览

---

## 已知边界(诚实)

- "皮层固化"目前是 dict(真实系统应蒸馏进大模型主干权重,LLM规模固化仍是学界空白)
- 实体抽取用轻量规则(产品化应上 NER/大模型)
- 服务无鉴权、单机内存态(生产级需加固:鉴权/并发锁/分布式)
- 小 LLM(如1.5b)可能不会好好利用注入的记忆,换大模型即改善

---

**一句话:任何语言、任何应用,一个 HTTP 请求,就能给自己的 LLM 加上"会持续学习 + 绝不幻觉"的记忆层。**
