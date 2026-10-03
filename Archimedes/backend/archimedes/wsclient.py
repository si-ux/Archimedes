"""Minimal RFC 6455 client — lets the CLI drive a running bridge."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
from urllib.parse import urlparse


class Client:
    def __init__(self, url: str = "ws://127.0.0.1:8791", timeout: float = 600.0):
        u = urlparse(url)
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 8791
        self.path = u.path or "/"
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._buf = b""

    # ------------------------------------------------------------ setup --
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {self.path} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(req.encode())
        while b"\r\n\r\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("handshake failed")
            self._buf += chunk
        head, self._buf = self._buf.split(b"\r\n\r\n", 1)
        if b"101" not in head.split(b"\r\n")[0]:
            raise ConnectionError(f"handshake rejected: {head.splitlines()[0]!r}")
        return self

    def close(self):
        if self.sock:
            try:
                self._frame(b"", 0x8)
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    # ------------------------------------------------------------ frames --
    def _need(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed by server")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _frame(self, payload: bytes, opcode: int = 0x1):
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        n = len(payload)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.sock.sendall(head + mask + masked)

    def send(self, obj: dict):
        self._frame(json.dumps(obj).encode("utf-8"))

    def recv(self) -> dict | None:
        chunks = []
        first = None
        while True:
            b0, b1 = self._need(2)
            fin, opcode, ln = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._need(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._need(8))[0]
            data = self._need(ln) if ln else b""
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._frame(data, 0xA)
                continue
            if opcode == 0xA:
                continue
            if first is None:
                first = opcode
            chunks.append(data)
            if fin:
                break
        return json.loads(b"".join(chunks).decode("utf-8"))


def unpack(field: dict):
    """Decode a base64 float32 field back into a numpy array."""
    import numpy as np
    raw = base64.b64decode(field["b64"])
    return np.frombuffer(raw, dtype=np.float32).reshape(field.get("shape") or (-1,))
