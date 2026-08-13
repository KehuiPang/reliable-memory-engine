#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memory_server 生产级加固端到端测试
  启动一个带鉴权+短自动save的服务子进程,验证:
    1) /health 免鉴权可探活
    2) 无 X-API-Key → 401;带正确 key → 通过
    3) learn / recall 正常
    4) 自动save生成存档目录
    5) 优雅关闭(SIGTERM)时把脏数据 flush 落盘,重启后 recall 仍命中
在 engine 目录下跑: python3 test_server.py
"""
import subprocess, time, os, json, signal, shutil, sys
import urllib.request

PORT = 8912
KEY = "test-secret-123"
SAVE_ROOT = "./_test_srv_mem"


def req(path, method="GET", body=None, key=KEY):
    url = "http://127.0.0.1:%d%s" % (PORT, path)
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    if key:
        r.add_header("X-API-Key", key)
    if data:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def start_server():
    env = dict(os.environ)
    env["MEM_API_KEY"] = KEY
    env["MEM_AUTOSAVE_SEC"] = "3"
    env["MEM_SAVE_ROOT"] = SAVE_ROOT
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.Popen([sys.executable, "memory_server.py", str(PORT)],
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # 等服务起来(轮询 health)
    for _ in range(60):
        try:
            s, _b = req("/health", key=None)
            if s == 200:
                return p
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("服务未能启动")


if os.path.exists(SAVE_ROOT):
    shutil.rmtree(SAVE_ROOT)

print("启动服务子进程...")
proc = start_server()
try:
    print("\n[测试1] /health 免鉴权")
    s, b = req("/health", key=None)
    print(" ", s, b)
    assert s == 200 and b["status"] == "ok" and b["auth"] is True

    print("\n[测试2] 鉴权拦截")
    s, b = req("/recall", "POST", {"query": "x"}, key=None)
    print("  无key →", s, b)
    assert s == 401
    s, b = req("/recall", "POST", {"query": "x"}, key="wrong")
    print("  错key →", s, b)
    assert s == 401

    print("\n[测试3] learn + recall(正确key)")
    s, b = req("/learn", "POST",
               {"tenant": "t1", "question": "figcheck何时上线",
                "answer": "2023年上线", "source": "user"})
    print("  learn →", s, b)
    assert s == 200 and b["ok"]
    s, b = req("/recall", "POST", {"tenant": "t1", "query": "figcheck什么时候上线的"})
    print("  recall →", s, b)
    assert s == 200 and b["decision"] == "answer" and "2023" in b["answer"]

    print("\n[测试4] 缺字段 → 400")
    s, b = req("/learn", "POST", {"tenant": "t1", "question": "只有问题"})
    print("  →", s, b)
    assert s == 400

    print("\n[测试5] 等待自动save(间隔3s)...")
    time.sleep(4.5)
    meta = os.path.join(SAVE_ROOT, "t1", "meta.json")
    print("  存档存在:", os.path.exists(meta))
    assert os.path.exists(meta), "自动save未生成存档"

    print("\n[测试6] 再写一条不save,SIGTERM 优雅关闭应flush")
    req("/learn", "POST",
        {"tenant": "t1", "question": "公司年会在哪", "answer": "三亚", "source": "user"})
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    print("  服务已关闭")

    print("\n[测试7] 重启后 recall 仍命中(证明flush落盘成功)")
    proc = start_server()
    s, b = req("/recall", "POST", {"tenant": "t1", "query": "公司年会在哪开"})
    print("  recall →", s, b)
    assert s == 200 and b["decision"] == "answer" and "三亚" in b["answer"], "关闭前的写入未落盘"

    print("\n" + "=" * 30)
    print("🎉 服务加固端到端测试全部通过")
finally:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    if os.path.exists(SAVE_ROOT):
        shutil.rmtree(SAVE_ROOT)
