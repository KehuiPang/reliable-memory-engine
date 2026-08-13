#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可靠记忆引擎 HTTP 服务 (多租户)
====================================
把引擎包成REST服务,让任意LLM/Agent通过HTTP挂载它(08文档"通用中立记忆层")。
- 标准库http.server,零额外依赖
- 多租户:不同应用/用户用tenant隔离,各自独立记忆
- 懒加载:租户首次使用时才建引擎(省内存;也确保faiss在torch后import)

接口:
  POST /learn    {tenant, question, answer, source?, paraphrases?}
  POST /recall   {tenant, query}
  POST /feedback {tenant, query, accepted?}
  POST /save     {tenant, path?}
  POST /load     {tenant, path?}
  GET  /stats?tenant=xxx
  GET  /tenants
启动: python3 memory_server.py [port]
"""
import sys, os, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ⚠️ 不在顶部import引擎(会连带torch);在首次建租户时import,确保faiss在torch后
_engines = {}          # tenant -> engine
_lock = threading.Lock()
_EngineClass = None
SAVE_ROOT = "./tenant_memory"


def _get_engine(tenant):
    """懒加载:租户首次使用时建引擎(线程安全)"""
    global _EngineClass
    with _lock:
        if _EngineClass is None:
            from indexed_engine import IndexedReliableMemory   # 此处才import(torch已可用)
            _EngineClass = IndexedReliableMemory
        if tenant not in _engines:
            # 若磁盘有该租户存档,自动load;否则新建
            path = os.path.join(SAVE_ROOT, tenant)
            if os.path.exists(os.path.join(path, "meta.json")):
                _engines[tenant] = _EngineClass.load(path)
            else:
                _engines[tenant] = _EngineClass()
        return _engines[tenant]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass   # 静音默认日志

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n == 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/tenants":
            return self._json(200, {"tenants": list(_engines.keys())})
        if u.path == "/stats":
            tenant = q.get("tenant", ["default"])[0]
            eng = _get_engine(tenant)
            return self._json(200, {"tenant": tenant, **eng.stats()})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        try:
            d = self._body()
        except Exception as e:
            return self._json(400, {"error": "bad json: %s" % e})
        tenant = d.get("tenant", "default")
        path = self.path

        if path == "/learn":
            eng = _get_engine(tenant)
            ok, msg = eng.learn(d["question"], d["answer"],
                                source=d.get("source", "user"),
                                paraphrases=d.get("paraphrases"))
            return self._json(200, {"ok": ok, "msg": msg})

        if path == "/recall":
            eng = _get_engine(tenant)
            ans, meta = eng.recall(d["query"])
            return self._json(200, {"answer": ans, "decision": meta["decision"],
                                    "meta": {k: (round(v, 3) if isinstance(v, float) else v)
                                             for k, v in meta.items()}})

        if path == "/feedback":
            eng = _get_engine(tenant)
            eng.feedback(d["query"], accepted=d.get("accepted", True))
            return self._json(200, {"ok": True})

        if path == "/save":
            eng = _get_engine(tenant)
            p = d.get("path", os.path.join(SAVE_ROOT, tenant))
            eng.save(p)
            return self._json(200, {"ok": True, "path": p})

        return self._json(404, {"error": "not found"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    os.makedirs(SAVE_ROOT, exist_ok=True)
    print("可靠记忆引擎服务启动 :%d (多租户,懒加载)" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
