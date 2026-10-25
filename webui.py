#!/usr/bin/env python3
"""
KohakuMoss Trainer WebUI
Dependency-free web server (stdlib only). Runs inside WSL; open http://localhost:7847
from the Windows browser. UI: webui_app/static, form schema: webui_app/schema.py.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from webui_app import configgen, schema  # noqa: E402
from webui_app.runner import RUNNER  # noqa: E402

HOST = "0.0.0.0"
# Note: Windows Hyper-V reserves 7756-7855 by default, which blocks WSL relay for 7847.
# 7878 falls in the safe open window (7856-7942) and avoids ComfyUI/Gradio 7860 conflicts.
PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 7878
STATIC = PROJECT_DIR / "webui_app" / "static"
PRESETS = PROJECT_DIR / "webui_app" / "presets"
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
QUIET_PATHS = ("/api/status", "/api/logs")


def to_windows_path(p: str) -> str:
    """/mnt/f/a/b -> F:\\a\\b for display; other paths unchanged."""
    if p.startswith("/mnt/") and len(p) > 6 and p[6] == "/":
        return f"{p[5].upper()}:\\" + p[7:].replace("/", "\\")
    return p


def scan_models(root: str, sub: str) -> list[str]:
    base = Path(configgen.to_linux_path(root)) / sub
    if not base.is_dir():
        return []
    found = sorted(base.rglob("*.safetensors"))
    return [to_windows_path(str(f)) for f in found]


def safe_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_. ").strip() or "preset"


class Handler(BaseHTTPRequestHandler):
    server_version = "KohakuMossUI"

    def log_message(self, fmt, *args):
        """Only report errors and non-polling requests (polling would flood the console)."""
        path = urlparse(self.path).path
        status = str(args[1]) if len(args) > 1 else ""
        if path in QUIET_PATHS or (status.startswith("2") and path.startswith("/static")):
            return
        if status.startswith(("4", "5")) and path == "/favicon.ico":
            return
        sys.stderr.write("[webui] %s %s\n" % (self.command, self.path))

    # --------------------------------------------------------------- helpers
    def send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path):
        if not path.is_file():
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            return {}

    # ------------------------------------------------------------------- GET
    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        path = url.path
        if path in ("/", "/index.html"):
            self.send_file(STATIC / "index.html")
        elif path.startswith("/static/"):
            target = (STATIC / unquote(path[len("/static/"):])).resolve()
            if STATIC.resolve() in target.parents:
                self.send_file(target)
            else:
                self.send_json({"error": "forbidden"}, 403)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path == "/api/schema":
            self.send_json({"tabs": schema.TABS})
        elif path == "/api/status":
            self.send_json(RUNNER.status())
        elif path == "/api/logs":
            self.send_json(RUNNER.read(int((q.get("since") or ["0"])[0])))
        elif path == "/api/scan":
            root = (q.get("root") or [""])[0]
            sub = (q.get("sub") or [""])[0]
            self.send_json({"files": scan_models(root, sub)})
        elif path == "/api/presets":
            PRESETS.mkdir(parents=True, exist_ok=True)
            self.send_json({"presets": sorted(p.stem for p in PRESETS.glob("*.json"))})
        elif path == "/api/preset":
            f = PRESETS / (safe_name((q.get("name") or [""])[0]) + ".json")
            self.send_json(json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {})
        elif path == "/api/last_probe":
            self.send_json(RUNNER.last_probe)
        else:
            self.send_json({"error": "not found"}, 404)

    # ------------------------------------------------------------------ POST
    def do_POST(self):
        path = urlparse(self.path).path
        data = self.body()
        if path == "/api/stop":
            RUNNER.stop()
            self.send_json({"ok": True})
        elif path == "/api/clear_logs":
            RUNNER.clear()
            self.send_json({"ok": True})
        elif path == "/api/doctor":
            ok = RUNNER.doctor()
            self.send_json({"ok": ok} if ok else {"error": "已有任务正在运行"}, 200 if ok else 400)
        elif path == "/api/probe":
            ok = RUNNER.probe(data)
            self.send_json({"ok": ok} if ok else {"error": "已有任务正在运行"}, 200 if ok else 400)
        elif path == "/api/train":
            if not configgen.merged(data)["datasets"]:
                self.send_json({"error": "请至少添加一个数据集目录"}, 400)
                return
            ok = RUNNER.train(data)
            self.send_json({"ok": ok} if ok else {"error": "已有任务正在运行"}, 200 if ok else 400)
        elif path == "/api/preview":
            spc = int(data.pop("_steps_per_epoch", 0) or RUNNER.last_probe.get("steps_per_epoch", 100))
            text, derived = configgen.build_config(data, spc)
            self.send_json({"config": text, "derived": derived})
        elif path == "/api/preset":
            PRESETS.mkdir(parents=True, exist_ok=True)
            name = safe_name(data.get("name", ""))
            (PRESETS / f"{name}.json").write_text(
                json.dumps(data.get("params", {}), ensure_ascii=False, indent=1), encoding="utf-8"
            )
            self.send_json({"ok": True, "name": name})
        elif path == "/api/preset_delete":
            f = PRESETS / (safe_name(data.get("name", "")) + ".json")
            if f.is_file():
                f.unlink()
            self.send_json({"ok": True})
        else:
            self.send_json({"error": "not found"}, 404)


def run_server():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"KohakuMoss WebUI running at http://localhost:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run_server()
