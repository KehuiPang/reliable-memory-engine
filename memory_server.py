#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可靠记忆引擎 HTTP 服务 (多租户 · 生产级加固版)
====================================================
把引擎包成REST服务,让任意LLM/Agent通过HTTP挂载它(08文档"通用中立记忆层")。
- 标准库http.server,零额外依赖
- 多租户:不同应用/用户用tenant隔离,各自独立记忆
- 懒加载:租户首次使用时才建引擎(省内存;也确保faiss在torch后import)

生产级加固:
  · 鉴权:可选 API Key(环境变量 MEM_API_KEY;设了才启用,请求头 X-API-Key)
  · 每租户写锁:learn/feedback/save 串行化(引擎非线程安全)
  · 定时自动 save:后台线程周期扫描,仅落盘有变更(脏)的租户
  · 健康检查:GET /health
  · 优雅关闭:SIGINT/SIGTERM 时先把所有脏租户存盘再退出
  · 输入校验:缺必填字段返回 400,内部异常返回 500 且不泄栈

接口:
  POST /learn    {tenant, question, answer, source?, paraphrases?}
  POST /recall   {tenant, query}
  POST /feedback {tenant, query, accepted?}
  POST /save     {tenant, path?}
  GET  /stats?tenant=xxx
  GET  /tenants
  GET  /health
启动: python3 memory_server.py [port]
环境变量:
  MEM_API_KEY        设置后所有请求须带 X-API-Key(/health 除外)
  MEM_AUTOSAVE_SEC   自动save间隔秒,默认60;设0关闭
  MEM_SAVE_ROOT      存档根目录,默认 ./tenant_memory
"""
import sys, os, json, threading, time, signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ⚠️ 不在顶部import引擎(会连带torch);在首次建租户时import,确保faiss在torch后
_engines = {}                    # tenant -> engine
_tenant_locks = {}               # tenant -> Lock(每租户一把写锁)
_global_lock = threading.Lock()  # 保护 _engines/_tenant_locks 结构本身
_EngineClass = None

API_KEY = os.environ.get("MEM_API_KEY", "").strip()
AUTOSAVE_SEC = int(os.environ.get("MEM_AUTOSAVE_SEC", "60"))
SAVE_ROOT = os.environ.get("MEM_SAVE_ROOT", "./tenant_memory")
_START_TS = time.time()
_stop_flag = threading.Event()


def _tenant_lock(tenant):
    with _global_lock:
        if tenant not in _tenant_locks:
            _tenant_locks[tenant] = threading.Lock()
        return _tenant_locks[tenant]


def _get_engine(tenant):
    """懒加载:租户首次使用时建引擎(线程安全)"""
    global _EngineClass
    with _global_lock:
        if _EngineClass is None:
            from indexed_engine import IndexedReliableMemory   # 此处才import(torch已可用)
            _EngineClass = IndexedReliableMemory
        if tenant not in _engines:
            path = os.path.join(SAVE_ROOT, tenant)
            if os.path.exists(os.path.join(path, "meta.json")):
                _engines[tenant] = _EngineClass.load(path)
            else:
                _engines[tenant] = _EngineClass()
        return _engines[tenant]


def _save_tenant(tenant, eng):
    p = os.path.join(SAVE_ROOT, tenant)
    with _tenant_lock(tenant):
        eng.save(p)
    return p


def _autosave_loop():
    """后台定时自动save:仅落盘有变更(_dirty)的租户"""
    if AUTOSAVE_SEC <= 0:
        return
    while not _stop_flag.wait(AUTOSAVE_SEC):
        try:
            items = list(_engines.items())
            saved = 0
            for tenant, eng in items:
                if getattr(eng, "_dirty", False):
                    _save_tenant(tenant, eng)
                    saved += 1
            if saved:
                print("[autosave] 已落盘 %d 个变更租户" % saved, flush=True)
        except Exception as e:
            print("[autosave] 出错(已跳过):%s" % e, flush=True)


def _flush_all(reason=""):
    """把所有脏租户存盘(优雅关闭/信号时调用)"""
    saved = 0
    for tenant, eng in list(_engines.items()):
        try:
            if getattr(eng, "_dirty", False):
                _save_tenant(tenant, eng)
                saved += 1
        except Exception as e:
            print("[flush] 租户 %s 存盘失败:%s" % (tenant, e), flush=True)
    print("[flush] %s 存盘 %d 个租户" % (reason, saved), flush=True)


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

    def _auth_ok(self):
        """鉴权:未设 API_KEY 则放行;设了则校验 X-API-Key"""
        if not API_KEY:
            return True
        return self.headers.get("X-API-Key", "") == API_KEY

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        # 健康检查不需要鉴权(供探活)
        if u.path == "/health":
            return self._json(200, {"status": "ok",
                                    "uptime_sec": round(time.time() - _START_TS, 1),
                                    "tenants": len(_engines),
                                    "auth": bool(API_KEY),
                                    "autosave_sec": AUTOSAVE_SEC})
        if not self._auth_ok():
            return self._json(401, {"error": "unauthorized"})
        if u.path == "/tenants":
            return self._json(200, {"tenants": list(_engines.keys())})
        if u.path == "/stats":
            tenant = q.get("tenant", ["default"])[0]
            try:
                eng = _get_engine(tenant)
                return self._json(200, {"tenant": tenant, **eng.stats()})
            except Exception as e:
                return self._json(500, {"error": "internal error"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._auth_ok():
            return self._json(401, {"error": "unauthorized"})
        try:
            d = self._body()
        except Exception as e:
            return self._json(400, {"error": "bad json"})
        if not isinstance(d, dict):
            return self._json(400, {"error": "body must be a json object"})
        tenant = d.get("tenant", "default")
        path = self.path

        try:
            if path == "/learn":
                if "question" not in d or "answer" not in d:
                    return self._json(400, {"error": "missing 'question' or 'answer'"})
                eng = _get_engine(tenant)
                with _tenant_lock(tenant):
                    ok, msg = eng.learn(d["question"], d["answer"],
                                        source=d.get("source", "user"),
                                        paraphrases=d.get("paraphrases"))
                return self._json(200, {"ok": ok, "msg": msg})

            if path == "/recall":
                if "query" not in d:
                    return self._json(400, {"error": "missing 'query'"})
                eng = _get_engine(tenant)
                # recall 只读,不加写锁(引擎 recall 不改结构,仅 hits+=1 可容忍竞态)
                ans, meta = eng.recall(d["query"])
                return self._json(200, {"answer": ans, "decision": meta["decision"],
                                        "meta": {k: (round(v, 3) if isinstance(v, float) else v)
                                                 for k, v in meta.items()}})

            if path == "/feedback":
                if "query" not in d:
                    return self._json(400, {"error": "missing 'query'"})
                eng = _get_engine(tenant)
                with _tenant_lock(tenant):
                    eng.feedback(d["query"], accepted=d.get("accepted", True))
                return self._json(200, {"ok": True})

            if path == "/save":
                eng = _get_engine(tenant)
                p = d.get("path")
                if p:
                    with _tenant_lock(tenant):
                        eng.save(p)
                else:
                    p = _save_tenant(tenant, eng)
                return self._json(200, {"ok": True, "path": p})

            return self._json(404, {"error": "not found"})
        except KeyError as e:
            return self._json(400, {"error": "missing field %s" % e})
        except Exception as e:
            print("[error] %s %s: %s" % (path, tenant, e), flush=True)
            return self._json(500, {"error": "internal error"})


def _install_signal_handlers(server):
    def _handler(signum, frame):
        print("\n[signal] 收到信号 %d,优雅关闭中..." % signum, flush=True)
        _stop_flag.set()
        _flush_all(reason="shutdown")
        # 在另一线程关服务,避免在信号处理里阻塞
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    os.makedirs(SAVE_ROOT, exist_ok=True)
    print("可靠记忆引擎服务启动 :%d (多租户,懒加载)" % port, flush=True)
    print("  鉴权: %s | 自动save: %s | 存档根: %s"
          % ("开(X-API-Key)" if API_KEY else "关",
             ("每%ds" % AUTOSAVE_SEC) if AUTOSAVE_SEC > 0 else "关",
             SAVE_ROOT), flush=True)
    t = threading.Thread(target=_autosave_loop, daemon=True)
    t.start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    _install_signal_handlers(server)
    try:
        server.serve_forever()
    finally:
        _flush_all(reason="exit")
