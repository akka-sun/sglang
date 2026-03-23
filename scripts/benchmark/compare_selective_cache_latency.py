#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from sglang_server_launcher import (
    SGLangServerConfig,
    SGLangServerHandle,
    build_sglang_command,
    start_sglang_server,
    stop_sglang_server,
)

# ===== Stable defaults / constants =====
DEFAULT_BASELINE_URL = "http://127.0.0.1:30000"
DEFAULT_SELECTIVE_URL = "http://127.0.0.1:30001"
DEFAULT_MODEL = "/shared/pii_detection/models/Qwen3-4B"
DEFAULT_BACKEND = "sglang-oai-chat"
DEFAULT_OUTPUT_DIR = "benchmark_results/selective_cache_user_isolation"

DEFAULT_SHAREGPT_INPUT_JSON = (
    "/shared/sglang/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_PROCESSED_USER_ONLY_JSON = (
    "/shared/sglang/datasets/sharegpt/ShareGPT_V3_user_only_all.json"
)

DEFAULT_SAMPLE_USER_COUNT = 110 # warmup + benchmark
DEFAULT_BENCHMARK_USER_COUNT = 100
DEFAULT_WARMUP_USERS = 10
DEFAULT_USER_SEED = 2026
DEFAULT_USER_TURN_MAX_TOKENS = 128
DEFAULT_REQUEST_RATE = 1.0
DEFAULT_MAX_CONCURRENCY = 8

DEFAULT_RANDOM_INPUT_LEN = 1024
DEFAULT_RANDOM_OUTPUT_LEN = 128
DEFAULT_SESSION_CAPACITY_OF_STR_LEN = 4_000_000
DEFAULT_SESSION_HTTP_TIMEOUT = 30.0
DEFAULT_SESSION_ID_PREFIX = "bench_user_"

DEFAULT_SGLANG_STARTUP_TIMEOUT = 900.0
DEFAULT_SGLANG_POLL_INTERVAL = 1.0
DEFAULT_SGLANG_LOG_DIRNAME = "sglang_logs"

# baseline: normal cache, selective: enable selective prefix cache
DEFAULT_BASELINE_SERVER_EXTRA_ARGS: List[str] = []
DEFAULT_SELECTIVE_SERVER_EXTRA_ARGS: List[str] = ["--enable-selective-prefix-cache"]


def remove_existing_file(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    raise IsADirectoryError(f"Expected file path but got directory: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run latency comparison with sequential SGLang lifecycle: "
            "baseline server first, then selective-prefix-cache server."
        )
    )
    parser.add_argument("--baseline-url", default=DEFAULT_BASELINE_URL)
    parser.add_argument("--selective-url", default=DEFAULT_SELECTIVE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default=DEFAULT_BACKEND)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--dataset-name", default="sharegpt-user-only")
    parser.add_argument("--prepare-sharegpt-user-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sharegpt-input-json", default=DEFAULT_SHAREGPT_INPUT_JSON)
    parser.add_argument(
        "--processed-user-only-json", default=DEFAULT_PROCESSED_USER_ONLY_JSON
    )

    parser.add_argument("--sample-user-count", type=int, default=DEFAULT_SAMPLE_USER_COUNT)
    parser.add_argument(
        "--benchmark-user-count", type=int, default=DEFAULT_BENCHMARK_USER_COUNT
    )
    parser.add_argument("--warmup-requests", type=int, default=DEFAULT_WARMUP_USERS)
    parser.add_argument("--sample-user-seed", type=int, default=DEFAULT_USER_SEED)
    parser.add_argument(
        "--user-turn-max-tokens", type=int, default=DEFAULT_USER_TURN_MAX_TOKENS
    )
    parser.add_argument("--request-rate", type=float, default=DEFAULT_REQUEST_RATE)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)

    parser.add_argument(
        "--prepared-openai-jsonl",
        default="",
        help="Optional custom path for merged OpenAI jsonl (non-isolation mode only).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to sglang.bench_serving.",
    )
    return parser.parse_args()


def read_last_jsonl(path: Path) -> Dict[str, Any]:
    last_line = ""
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                last_line = line
    if not last_line:
        raise RuntimeError(f"No JSON records in {path}")
    return json.loads(last_line)


def run_one(
    label: str,
    base_url: str,
    args: argparse.Namespace,
    output_dir: Path,
    run_extra_args: Optional[List[str]] = None,
) -> Dict[str, Any]:
    output_file = output_dir / f"{label}.jsonl"
    remove_existing_file(output_file)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        args.backend,
        "--base-url",
        base_url.rstrip("/"),
        "--model",
        args.model,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(args.max_concurrency),
        "--random-input-len",
        str(DEFAULT_RANDOM_INPUT_LEN),
        "--random-output-len",
        str(DEFAULT_RANDOM_OUTPUT_LEN),
        "--output-file",
        str(output_file),
    ]

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd.extend(extra_args)
    if run_extra_args:
        cmd.extend(run_extra_args)

    repo_python_dir = Path(__file__).resolve().parents[2] / "python"
    env = os.environ.copy()
    if repo_python_dir.exists():
        prev_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(repo_python_dir)
            if not prev_pythonpath
            else str(repo_python_dir) + os.pathsep + prev_pythonpath
        )
        print(f"[run] {label}: prepend PYTHONPATH={repo_python_dir}", flush=True)

    print(f"[run] {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    metrics = read_last_jsonl(output_file)
    metrics["label"] = label
    return metrics


def pct_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value * 100.0


def build_report(baseline: Dict[str, Any], selective: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "request_throughput",
        "output_throughput",
        "total_throughput",
    ]
    deltas = {}
    for key in keys:
        b = float(baseline.get(key, 0.0))
        s = float(selective.get(key, 0.0))
        deltas[key] = {
            "baseline": b,
            "selective": s,
            "delta_pct": pct_change(s, b),
        }
    return {"baseline": baseline, "selective": selective, "deltas": deltas}


def print_table(report: Dict[str, Any]) -> None:
    rows = [
        ("mean_ttft_ms", "Lower better"),
        ("p99_ttft_ms", "Lower better"),
        ("mean_e2e_latency_ms", "Lower better"),
        ("p99_e2e_latency_ms", "Lower better"),
        ("request_throughput", "Higher better"),
        ("output_throughput", "Higher better"),
    ]
    print("\n=== Selective Prefix Cache Latency Comparison ===")
    print(f"{'Metric':30} {'Baseline':>12} {'Selective':>12} {'Delta%':>10}  {'Direction'}")
    for key, direction in rows:
        item = report["deltas"][key]
        print(
            f"{key:30} "
            f"{item['baseline']:12.2f} "
            f"{item['selective']:12.2f} "
            f"{item['delta_pct']:10.2f}  {direction}"
        )


def normalize_turn(turn: Dict[str, Any]) -> Tuple[str, str]:
    role_raw = (turn.get("from") or turn.get("role") or "").strip().lower()
    content = turn.get("value", turn.get("content", "")) or ""
    if role_raw in ("human", "user"):
        return "user", content
    if role_raw in ("gpt", "assistant"):
        return "assistant", content
    return "", content


def preprocess_sharegpt_user_only(input_path: Path, output_path: Path) -> Dict[str, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"ShareGPT file not found: {input_path}")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    user_turns: Dict[str, List[str]] = {}
    total_examples = 0
    total_kept_turns = 0

    for item in data:
        total_examples += 1
        user_id = str(item.get("id", "")).strip()
        if not user_id:
            continue
        turns = item.get("conversations", item.get("conversation", []))
        if not isinstance(turns, list):
            continue

        kept: List[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            role, content = normalize_turn(turn)
            text = content.strip()
            if role == "user" and text:
                kept.append(text)

        if not kept:
            continue
        user_turns.setdefault(user_id, []).extend(kept)
        total_kept_turns += len(kept)

    output_rows = [
        {"user_id": user_id, "turns": turns}
        for user_id, turns in user_turns.items()
        if turns
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_existing_file(output_path)
    output_path.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "input_examples": total_examples,
        "output_users": len(output_rows),
        "output_turns": total_kept_turns,
    }


def load_user_only_data(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed user-only file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, List[str]] = {}
    for item in data:
        user_id = str(item.get("user_id", "")).strip()
        turns = item.get("turns", [])
        if not user_id or not isinstance(turns, list):
            continue
        cleaned_turns = [str(x).strip() for x in turns if str(x).strip()]
        if cleaned_turns:
            result[user_id] = cleaned_turns
    return result


def to_session_id(prefix: str, user_id: str) -> str:
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{digest}"


def sample_user_ids(
    user_turns: Dict[str, List[str]], sample_user_count: int, sample_user_seed: int
) -> List[str]:
    all_user_ids = list(user_turns.keys())
    if not all_user_ids:
        raise RuntimeError("No valid users in processed user-only dataset.")
    random.seed(sample_user_seed)
    return random.sample(all_user_ids, min(sample_user_count, len(all_user_ids)))


def build_openai_requests_from_user_only(
    user_turns: Dict[str, List[str]],
    selected_ids: List[str],
    max_tokens: int,
    include_isolation_fields: bool,
    output_path: Path,
) -> Tuple[int, int, List[str]]:
    if not selected_ids:
        raise RuntimeError("No sampled users for benchmark dataset.")

    rows: List[Dict[str, Any]] = []
    session_ids: List[str] = []
    for user_id in selected_ids:
        sid = to_session_id(DEFAULT_SESSION_ID_PREFIX, user_id)
        row: Dict[str, Any] = {
            "user_turns": list(user_turns[user_id]),
            "max_tokens": max_tokens,
        }
        if include_isolation_fields:
            session_ids.append(sid)
            row["extra_key"] = user_id
            row["session_params"] = {"id": sid}
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    remove_existing_file(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_turns = sum(len(user_turns[user_id]) for user_id in selected_ids)
    return len(selected_ids), total_turns, session_ids


def recreate_sessions(base_url: str, session_ids: Sequence[str]) -> None:
    close_url = base_url.rstrip("/") + "/close_session"
    open_url = base_url.rstrip("/") + "/open_session"

    for i, session_id in enumerate(session_ids, start=1):
        try:
            requests.post(
                close_url,
                json={"session_id": session_id},
                timeout=DEFAULT_SESSION_HTTP_TIMEOUT,
            )
        except Exception:
            pass

        response = requests.post(
            open_url,
            json={
                "capacity_of_str_len": DEFAULT_SESSION_CAPACITY_OF_STR_LEN,
                "session_id": session_id,
            },
            timeout=DEFAULT_SESSION_HTTP_TIMEOUT,
        )
        response.raise_for_status()

        if i % 200 == 0:
            print(f"[session] {base_url} opened {i}/{len(session_ids)}", flush=True)


def ensure_extra_arg(args: argparse.Namespace, name: str, value: str) -> None:
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    if name not in extra_args:
        extra_args.extend([name, value])
    args.extra_args = extra_args


def ensure_sglang_backend(backend: str) -> None:
    if not backend.startswith("sglang"):
        raise ValueError(
            f"Current script now manages SGLang lifecycle only. "
            f"Please use sglang backend, got: {backend}"
        )


def start_managed_sglang(
    label: str,
    base_url: str,
    model_path: str,
    output_dir: Path,
    selective_mode: bool,
) -> SGLangServerHandle:
    log_dir = output_dir / DEFAULT_SGLANG_LOG_DIRNAME
    extra_args = (
        DEFAULT_SELECTIVE_SERVER_EXTRA_ARGS
        if selective_mode
        else DEFAULT_BASELINE_SERVER_EXTRA_ARGS
    )
    config = SGLangServerConfig(
        base_url=base_url,
        model_path=model_path,
        python_executable=sys.executable,
        extra_args=list(extra_args),
        startup_timeout=DEFAULT_SGLANG_STARTUP_TIMEOUT,
        poll_interval=DEFAULT_SGLANG_POLL_INTERVAL,
        log_file=log_dir / f"{label}.log",
    )
    print(f"[sglang] start {label}: {' '.join(build_sglang_command(config))}", flush=True)
    handle = start_sglang_server(config)
    print(f"[sglang] ready {label}: {base_url}", flush=True)
    return handle


def main() -> int:
    args = parse_args()
    ensure_sglang_backend(args.backend)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_user_only_path = Path(args.processed_user_only_json)
    need_user_only_flow = (
        args.prepare_sharegpt_user_only
        or args.prepare_only
        or args.dataset_name == "sharegpt-user-only"
    )

    if need_user_only_flow:
        if args.prepare_sharegpt_user_only or not processed_user_only_path.exists():
            stats = preprocess_sharegpt_user_only(
                input_path=Path(args.sharegpt_input_json),
                output_path=processed_user_only_path,
            )
            print(
                "[prepare] done "
                f"examples={stats['input_examples']} users={stats['output_users']} "
                f"turns={stats['output_turns']} output={processed_user_only_path}",
                flush=True,
            )
        if args.prepare_only:
            return 0

    baseline_run_extra_args: List[str] = []
    selective_run_extra_args: List[str] = []
    selective_session_ids: List[str] = []

    if args.dataset_name == "sharegpt-user-only":
        user_turns = load_user_only_data(processed_user_only_path)
        sampled_user_ids = sample_user_ids(
            user_turns=user_turns,
            sample_user_count=args.sample_user_count,
            sample_user_seed=args.sample_user_seed,
        )
        sampled_users = len(sampled_user_ids)

        if args.benchmark_user_count <= 0:
            raise ValueError("--benchmark-user-count must be > 0.")
        if args.warmup_requests < 0:
            raise ValueError("--warmup-requests must be >= 0.")

        needed_users = args.benchmark_user_count + args.warmup_requests
        if sampled_users < needed_users:
            raise ValueError(
                f"Need at least {needed_users} sampled users, but only got {sampled_users}. "
                "Increase --sample-user-count."
            )
        selected_ids = sampled_user_ids[:needed_users]

        baseline_dataset_path = (
            output_dir
            / f"sharegpt_user_only_openai_baseline_shared_seed{args.sample_user_seed}.jsonl"
        )
        selective_dataset_path = (
            output_dir
            / f"sharegpt_user_only_openai_selective_isolated_seed{args.sample_user_seed}.jsonl"
        )

        _, baseline_turns, _ = build_openai_requests_from_user_only(
            user_turns=user_turns,
            selected_ids=selected_ids,
            max_tokens=args.user_turn_max_tokens,
            include_isolation_fields=False,
            output_path=baseline_dataset_path,
        )
        _, selective_turns, selective_session_ids = build_openai_requests_from_user_only(
            user_turns=user_turns,
            selected_ids=selected_ids,
            max_tokens=args.user_turn_max_tokens,
            include_isolation_fields=True,
            output_path=selective_dataset_path,
        )

        print(
            f"[prepare] sampled_users={sampled_users} selected_users={needed_users} "
            f"warmup_users={args.warmup_requests} benchmark_users={args.benchmark_user_count} "
            f"baseline_turns={baseline_turns} selective_turns={selective_turns} "
            f"baseline_dataset={baseline_dataset_path} selective_dataset={selective_dataset_path}",
            flush=True,
        )

        args.dataset_name = "openai"
        args.num_prompts = needed_users
        baseline_run_extra_args = [
            "--dataset-path",
            str(baseline_dataset_path),
            "--seed",
            str(args.sample_user_seed),
        ]
        selective_run_extra_args = [
            "--dataset-path",
            str(selective_dataset_path),
            "--seed",
            str(args.sample_user_seed),
        ]

    ensure_extra_arg(args, "--warmup-requests", str(args.warmup_requests))

    baseline: Dict[str, Any]
    selective: Dict[str, Any]
    baseline_server: Optional[SGLangServerHandle] = None
    selective_server: Optional[SGLangServerHandle] = None

    try:
        baseline_server = start_managed_sglang(
            label="baseline",
            base_url=args.baseline_url,
            model_path=args.model,
            output_dir=output_dir,
            selective_mode=False,
        )
        baseline = run_one(
            label="baseline",
            base_url=args.baseline_url,
            args=args,
            output_dir=output_dir,
            run_extra_args=baseline_run_extra_args,
        )
    finally:
        stop_sglang_server(baseline_server)
        print("[sglang] stopped baseline", flush=True)

    try:
        selective_server = start_managed_sglang(
            label="selective",
            base_url=args.selective_url,
            model_path=args.model,
            output_dir=output_dir,
            selective_mode=True,
        )
        if selective_session_ids:
            print("[session] preparing selective sessions...", flush=True)
            recreate_sessions(args.selective_url, selective_session_ids)
        selective = run_one(
            label="selective",
            base_url=args.selective_url,
            args=args,
            output_dir=output_dir,
            run_extra_args=selective_run_extra_args,
        )
    finally:
        stop_sglang_server(selective_server)
        print("[sglang] stopped selective", flush=True)

    report = build_report(baseline, selective)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comparison_{timestamp}.json"
    remove_existing_file(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_table(report)
    print(f"\nSaved comparison report to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
