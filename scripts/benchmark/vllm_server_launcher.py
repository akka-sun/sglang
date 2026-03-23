#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import requests


def parse_host_port(base_url: str) -> Tuple[str, int]:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"Invalid base url: {base_url}. Expected format like http://127.0.0.1:30000"
        )
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    return host, port


@dataclass
class VLLMServerConfig:
    base_url: str
    model: str
    python_executable: str = sys.executable
    module: str = "vllm.entrypoints.openai.api_server"
    served_model_name: Optional[str] = None
    extra_args: Optional[List[str]] = None
    startup_timeout: float = 600.0
    poll_interval: float = 1.0
    log_file: Optional[Path] = None


@dataclass
class VLLMServerHandle:
    config: VLLMServerConfig
    process: subprocess.Popen
    log_fp: Optional[object] = None


def build_vllm_command(config: VLLMServerConfig) -> List[str]:
    host, port = parse_host_port(config.base_url)
    cmd: List[str] = [
        config.python_executable,
        "-m",
        config.module,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        config.model,
    ]
    if config.served_model_name:
        cmd.extend(["--served-model-name", config.served_model_name])
    if config.extra_args:
        cmd.extend(config.extra_args)
    return cmd


def wait_until_ready(
    base_url: str,
    process: subprocess.Popen,
    timeout_sec: float,
    poll_interval_sec: float,
) -> None:
    deadline = time.time() + timeout_sec
    models_url = base_url.rstrip("/") + "/v1/models"
    last_error: Optional[str] = None
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited early (returncode={process.returncode})."
            )
        try:
            response = requests.get(models_url, timeout=5.0)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(max(0.1, poll_interval_sec))
    raise TimeoutError(
        f"Timed out waiting for vLLM ready at {models_url}. Last error: {last_error}"
    )


def start_vllm_server(config: VLLMServerConfig) -> VLLMServerHandle:
    cmd = build_vllm_command(config)
    log_fp = None
    stdout = None
    stderr = None
    if config.log_file is not None:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        if config.log_file.exists():
            if config.log_file.is_file() or config.log_file.is_symlink():
                config.log_file.unlink()
            else:
                raise IsADirectoryError(
                    f"Expected log file path but got directory: {config.log_file}"
                )
        log_fp = config.log_file.open("w", encoding="utf-8")
        stdout = log_fp
        stderr = subprocess.STDOUT

    process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
    handle = VLLMServerHandle(config=config, process=process, log_fp=log_fp)
    wait_until_ready(
        base_url=config.base_url,
        process=process,
        timeout_sec=config.startup_timeout,
        poll_interval_sec=config.poll_interval,
    )
    return handle


def stop_vllm_server(handle: Optional[VLLMServerHandle], grace_period_sec: float = 10.0) -> None:
    if handle is None:
        return
    proc = handle.process
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace_period_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=grace_period_sec)
    finally:
        if handle.log_fp is not None:
            handle.log_fp.close()


def _parse_extra_args(extra_args: str) -> List[str]:
    extra_args = (extra_args or "").strip()
    if not extra_args:
        return []
    return shlex.split(extra_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch one vLLM OpenAI server and keep it running until interrupted."
    )
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:30000")
    parser.add_argument("--model", required=True, help="Model path or name")
    parser.add_argument("--served-model-name", default="", help="Optional served model name")
    parser.add_argument("--startup-timeout", type=float, default=600.0, help="Seconds to wait server ready")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval seconds")
    parser.add_argument("--log-file", default="", help="Optional log file path")
    parser.add_argument(
        "--extra-args",
        default="",
        help='Extra args passed to vLLM server, e.g. "--tensor-parallel-size 1"',
    )
    args = parser.parse_args()

    config = VLLMServerConfig(
        base_url=args.base_url,
        model=args.model,
        served_model_name=args.served_model_name or None,
        extra_args=_parse_extra_args(args.extra_args),
        startup_timeout=args.startup_timeout,
        poll_interval=args.poll_interval,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    print(f"[vllm] starting: {' '.join(build_vllm_command(config))}", flush=True)
    handle = start_vllm_server(config)
    print(f"[vllm] ready at {config.base_url}", flush=True)
    try:
        while True:
            if handle.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited unexpectedly (returncode={handle.process.returncode})"
                )
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[vllm] interrupted, shutting down...", flush=True)
    finally:
        stop_vllm_server(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
