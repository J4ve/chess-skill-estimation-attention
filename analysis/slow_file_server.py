#!/usr/bin/env python3
"""Throttled, Range-aware HTTP file server for the pipeline recovery test.

The recovery test has to kill a worker in the MIDDLE of a download and then
prove the next run resumes rather than restarting. That needs a download slow
enough to interrupt and a server that honours `Range:`, which is exactly what
`curl -C -` sends. Serving from a plain file:// URL would test neither.

Every Range request is appended to <root>/.range_log so the test can assert that
a resume actually happened instead of taking the client's word for it.

    python slow_file_server.py --root DIR --port 8099 --bytes-per-sec 300000
"""

import argparse
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHUNK = 16384


class Handler(BaseHTTPRequestHandler):
    root = "."
    rate = 300_000

    def log_message(self, *_):  # keep the test output readable
        pass

    def _resolve(self):
        name = os.path.basename(self.path)
        path = os.path.join(self.root, name)
        return path if os.path.isfile(path) else None

    def do_HEAD(self):
        path = self._resolve()
        if not path:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        path = self._resolve()
        if not path:
            self.send_error(404)
            return
        size = os.path.getsize(path)

        start = 0
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d+)-", rng)
            if m:
                start = int(m.group(1))
            with open(os.path.join(self.root, ".range_log"), "a") as f:
                f.write(f"{os.path.basename(path)} {rng}\n")
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        else:
            self.send_response(200)

        self.send_header("Content-Length", str(size - start))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        sent = 0
        began = time.monotonic()
        with open(path, "rb") as f:
            f.seek(start)
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    return  # the client was killed; that is the point of the test
                sent += len(buf)
                target = began + sent / self.rate
                now = time.monotonic()
                if target > now:
                    time.sleep(target - now)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--bytes-per-sec", type=int, default=300_000)
    args = ap.parse_args()

    Handler.root = args.root
    Handler.rate = args.bytes_per_sec
    # Threading matters: a single-threaded server serialises concurrent workers,
    # and a slow transfer then blocks every other request long enough for curl's
    # own --retry to fire, which appends a second copy of the range onto the
    # output file. That is a server artifact, not pipeline behaviour, and it
    # would otherwise show up as a bogus test failure.
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
