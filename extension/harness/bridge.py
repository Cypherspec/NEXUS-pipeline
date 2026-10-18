"""A minimal stand-in for the Mantis backend bridge.

Mantis loads `backend/main.py` and calls `invoke(action, payload, ctx)`. This
serves exactly that over HTTP so the built panel bundle can be driven outside
Mantis -- same backend module, same actions, same payloads. It exists to
screenshot and exercise the real panel, not to replace the host: nothing here
is part of the installed extension.

    python -m extension.harness.bridge --port 8770
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # extension/
BACKEND = ROOT / "backend" / "main.py"

spec = importlib.util.spec_from_file_location("nexus_extension_backend", BACKEND)
backend = importlib.util.module_from_spec(spec)
sys.modules["nexus_extension_backend"] = backend
spec.loader.exec_module(backend)

CTX = {"action": "", "payload": {}, "project_id": "harness",
       "user_id": "harness", "extension_id": "nexus.drug-discovery",
       "extension_version": "0.2.0"}


class Handler(BaseHTTPRequestHandler):
    """Routes POST /invoke into the extension backend."""

    def _send(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:                       # noqa: N802
        self._send(204, {})

    def do_POST(self) -> None:                          # noqa: N802
        if self.path.rstrip("/") != "/invoke":
            self._send(404, {"error": "only /invoke"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
            action = request.get("action")
            result = backend.invoke(action, request.get("payload") or {},
                                    {**CTX, "action": action})
            self._send(200, {"ok": True, "result": result})
        except Exception as exc:                        # noqa: BLE001
            self._send(400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, *_args) -> None:
        """Quiet: the harness output is the screenshot run, not a request log."""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"extension backend bridge on :{args.port} "
          f"(actions: {', '.join(sorted(backend._ACTIONS))})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
