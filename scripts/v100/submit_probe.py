"""Send v100_probe.py to a V100 node through the Looma orchestrator, and follow it.

    export LOOMA_ADMIN_TOKEN=...            # the orchestrator's admin token
    python scripts/v100/submit_probe.py submit [--node <id>] [-- probe args...]
    python scripts/v100/submit_probe.py logs <task_id> [--tail 300]
    python scripts/v100/submit_probe.py state <task_id>

Goes through POST /admin/tasks with its own `environment`, so the default FreeToken
pin and every other node stay untouched. The torch lines are bare wheel URLs on
purpose: the agent installs anything it recognises as `torch...` from the cu1xx
index it picks for the driver (cu128 here, which has no sm_70), while a bare URL
goes through the plain pass and pulls its nvidia-*-cu12 dependencies from PyPI.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_ENGINE = ROOT.parent / "looma" / "payloads" / "looma_freetoken" / "looma_freetoken" / "engine.py"
RELEASE = "https://github.com/LoomaFloat/freetoken/releases/download/{tag}/"
TORCH = "https://download.pytorch.org/whl/cu126/"


def requirements(tag: str, version: str) -> list[str]:
    base = RELEASE.format(tag=tag)
    local = version.split("+", 1)[1]
    return [
        f"{TORCH}torch-2.11.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl",
        f"{TORCH}torchvision-0.26.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl",
        f"freetoken @ {base}freetoken-0.1.3%2B{local}-cp312-cp312-linux_x86_64.whl",
        f"{RELEASE.format(tag='v0.1.3-looma25-v100.1')}"
        "freetoken_kernel_cache-0.1.3%2Bcu126.looma25.v100.1-py3-none-linux_x86_64.whl",
        "ziglang==0.16.0",
        "pytest>=8,<9",
    ]


def call(method: str, path: str, body: dict | None = None) -> dict:
    url = os.environ.get("LOOMA_URL", "https://loomafloat.ru").rstrip("/") + path
    token = os.environ.get("LOOMA_ADMIN_TOKEN")
    if not token:
        sys.exit("set LOOMA_ADMIN_TOKEN (and LOOMA_URL if not https://loomafloat.ru)")
    request = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", "X-Looma-Admin-Token": token})
    try:
        with urllib.request.urlopen(request, timeout=120) as answer:
            return json.loads(answer.read())
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method} {path}: {exc.code} {exc.read().decode(errors='replace')[:500]}")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def tests_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(ROOT / "tests", arcname="tests",
                    filter=lambda info: None if "__pycache__" in info.name else info)
    return buffer.getvalue()


def find_v100() -> str:
    nodes = call("GET", "/admin/agents").get("nodes") or []
    hits = [n for n in nodes if "V100" in (n.get("gpu_name") or "")]
    for n in nodes:
        print(f"  {n.get('node_id')}: {n.get('gpu_name')} cuda {n.get('cuda_version')} "
              f"gpus free {n.get('gpus_free')}/{n.get('gpus_total')}")
    if len(hits) != 1:
        sys.exit(f"expected exactly one V100 node, found {len(hits)}; pass --node")
    return hits[0]["node_id"]


def submit(args) -> None:
    version = {}
    exec((ROOT / "python" / "freetoken" / "version.py").read_text(), version)
    node = args.node or find_v100()
    reqs = requirements(args.tag, version["__version__"])
    body = {
        "node_id": node,
        "command": ["python", "v100_probe.py", *args.probe_args],
        "environment": {"kind": "python", "requirements": reqs},
        "resources": {"gpus": 1, "cpus": args.cpus},
        "timeout_s": int(args.hours * 3600),
        "env": {"PYTHONUNBUFFERED": "1"},
        "inputs": {
            "v100_probe.py": b64((ROOT / "scripts" / "v100" / "v100_probe.py").read_bytes()),
            "looma_ft_engine.py": b64(PAYLOAD_ENGINE.read_bytes()),
            "tests.tar.gz": b64(tests_archive()),
        },
    }
    print(f"node {node}, requirements:")
    for line in reqs:
        print(f"  {line}")
    record = call("POST", "/admin/tasks", body)
    print(json.dumps({k: record.get(k) for k in ("task_id", "state", "node_id")}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--node", default="")
    s.add_argument("--tag", default="v0.1.3-looma25-v100.2")
    s.add_argument("--cpus", type=float, default=16)
    s.add_argument("--hours", type=float, default=4)
    s.add_argument("probe_args", nargs=argparse.REMAINDER)
    lg = sub.add_parser("logs")
    lg.add_argument("task_id")
    lg.add_argument("--tail", type=int, default=300)
    st = sub.add_parser("state")
    st.add_argument("task_id")
    args = parser.parse_args()
    if args.cmd == "submit":
        args.probe_args = [a for a in args.probe_args if a != "--"]
        submit(args)
    elif args.cmd == "logs":
        got = call("GET", f"/admin/tasks/{args.task_id}/logs?tail={args.tail}")
        print(got.get("text") or json.dumps(got, ensure_ascii=False, indent=1))
    else:
        print(json.dumps(call("GET", f"/admin/tasks/{args.task_id}"), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
