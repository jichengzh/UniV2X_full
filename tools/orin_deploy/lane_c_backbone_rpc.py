#!/usr/bin/env python3
"""Localhost-only RPC transport for an Orin TensorRT multiscale backbone."""

from __future__ import annotations

import argparse
import io
import ipaddress
import json
import socket
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np


HEADER = struct.Struct("!Q")
DEFAULT_MAX_PAYLOAD = 512 * 1024 * 1024


def validate_bind_host(host: str) -> None:
    address = ipaddress.ip_address(host)
    if not address.is_loopback:
        raise ValueError("RPC bind must remain on a loopback address")


def encode_array_bundle(arrays: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
    return zlib.compress(buffer.getvalue(), level=1)


def decode_array_bundle(
    payload: bytes, *, max_uncompressed_bytes: int = DEFAULT_MAX_PAYLOAD
) -> dict[str, np.ndarray]:
    if len(payload) > max_uncompressed_bytes:
        raise ValueError("compressed payload exceeds configured limit")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(payload, max_uncompressed_bytes + 1)
    if len(raw) > max_uncompressed_bytes or decompressor.unconsumed_tail:
        raise ValueError("uncompressed payload exceeds configured limit")
    raw += decompressor.flush()
    if len(raw) > max_uncompressed_bytes:
        raise ValueError("uncompressed payload exceeds configured limit")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("RPC peer closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(connection: socket.socket, payload: bytes) -> None:
    connection.sendall(HEADER.pack(len(payload)))
    connection.sendall(payload)


def receive_packet(
    connection: socket.socket, *, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD
) -> bytes:
    size = HEADER.unpack(_recv_exact(connection, HEADER.size))[0]
    if size > max_payload_bytes:
        raise ValueError(f"RPC payload {size} exceeds limit {max_payload_bytes}")
    return _recv_exact(connection, int(size))


class OrinBackboneRpcClient:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.connection = socket.create_connection((host, port), timeout_seconds)
        self.connection.settimeout(timeout_seconds)
        self.call_count = 0
        self.request_bytes = 0
        self.response_bytes = 0
        self.elapsed_seconds = 0.0

    def __call__(self, spatial_features: np.ndarray) -> list[tuple[str, np.ndarray]]:
        request = encode_array_bundle(
            {"spatial_features": np.asarray(spatial_features, dtype=np.float32)}
        )
        started = time.perf_counter()
        send_packet(self.connection, request)
        response = receive_packet(self.connection)
        elapsed = time.perf_counter() - started
        outputs = decode_array_bundle(response)
        self.call_count += 1
        self.request_bytes += len(request)
        self.response_bytes += len(response)
        self.elapsed_seconds += elapsed
        return list(outputs.items())

    def close(self) -> None:
        try:
            send_packet(self.connection, encode_array_bundle({"__shutdown__": np.ones(1)}))
        except (EOFError, OSError):
            pass
        self.connection.close()

    def audit(self) -> dict[str, Any]:
        return {
            "transport": "SSH_local_forward_to_orin_loopback",
            "call_count": self.call_count,
            "compressed_request_bytes": self.request_bytes,
            "compressed_response_bytes": self.response_bytes,
            "rpc_elapsed_seconds": self.elapsed_seconds,
        }


def _write_server_audit(path: Path, audit: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def serve(args: argparse.Namespace) -> int:
    from lane_c_backbone_parity_runner import _deserialize_engine, _prepare_execution

    _, engine = _deserialize_engine(args.engine)
    validate_bind_host(args.host)
    audit: dict[str, Any] = {
        "schema_version": "lane_c_orin_backbone_rpc_server_v1",
        "bind_host": args.host,
        "bind_port": args.port,
        "loopback_only": ipaddress.ip_address(args.host).is_loopback,
        "request_count": 0,
        "request_compressed_bytes": 0,
        "response_compressed_bytes": 0,
        "engine_path": str(args.engine),
        "status": "starting",
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        audit["status"] = "listening"
        _write_server_audit(args.audit_json, audit)
        connection, peer = listener.accept()
        with connection:
            audit["peer"] = list(peer)
            audit["status"] = "serving"
            _write_server_audit(args.audit_json, audit)
            while True:
                try:
                    request_payload = receive_packet(connection)
                except EOFError:
                    break
                request = decode_array_bundle(request_payload)
                if "__shutdown__" in request:
                    break
                if set(request) != {"spatial_features"}:
                    raise ValueError(f"unexpected RPC request fields: {sorted(request)}")
                spatial = np.asarray(request["spatial_features"], dtype=np.float32)
                if tuple(spatial.shape) != (2, 64, 128, 256):
                    raise ValueError(f"unexpected RPC input shape: {list(spatial.shape)}")
                execute, stream, tensors, output_names = _prepare_execution(engine, spatial)
                execute(False)
                stream.synchronize()
                response_arrays = {
                    name: tensors[name].detach().cpu().float().numpy()
                    for name in output_names
                }
                response_payload = encode_array_bundle(response_arrays)
                send_packet(connection, response_payload)
                audit["request_count"] += 1
                audit["request_compressed_bytes"] += len(request_payload)
                audit["response_compressed_bytes"] += len(response_payload)
                if audit["request_count"] % 25 == 0:
                    _write_server_audit(args.audit_json, audit)
    audit["status"] = "complete"
    _write_server_audit(args.audit_json, audit)
    return 0


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=_path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--audit-json", type=_path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return serve(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
