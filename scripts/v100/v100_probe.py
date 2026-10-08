"""Does the looma/v100 build actually run on a Tesla V100? One Looma task answers it.

Ships as a task input next to `looma_ft_engine.py` (a copy of
looma/payloads/looma_freetoken/looma_freetoken/engine.py, for `provide_compiler`)
and `tests.tar.gz` (this repo's tests/). Stages, cheapest first, each one logged
so a failure says where it broke:

  1. what torch / triton / freetoken this env really got, and whether torch has sm_70;
  2. a C compiler for Triton (the agent image has none; zig cc via ziglang);
  3. every prebuilt kernel-cache module loads on this card;
  4. bf16 / fp16 matmul and the activation kernels against a torch reference;
  5. the repo's GPU tests for the listed directories;
  6. ft bench bw + ft serve on a real MoE checkpoint, one request, tokens per second.

    python v100_probe.py --model Qwen/Qwen3-30B-A3B [--serve-arg=--dtype=float16]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request

FT = (sys.executable, "-m", "freetoken.cli")


def say(text: str = "") -> None:
    print(text, flush=True)


def stage(title: str) -> None:
    say(f"\n===== {title} =====")


def run(*command: str, timeout: float = 900) -> tuple[int, str]:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, f"no command {command[0]!r}"
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return 124, out + f"\n{command[0]} timed out after {timeout:.0f} s"
    return done.returncode, (done.stdout or "") + (done.stderr or "")


def card_memory() -> str:
    code, text = run("nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader", timeout=30)
    return text.strip().splitlines()[0] if code == 0 and text.strip() else "?"


def environment() -> bool:
    stage("1. environment")
    say(f"python {sys.version.split()[0]} at {sys.executable}")
    code, text = run("nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total",
                     "--format=csv,noheader", timeout=30)
    say(f"nvidia-smi: {text.strip()}")
    import torch

    say(f"torch {torch.__version__}, cuda {torch.version.cuda}, available {torch.cuda.is_available()}")
    arches = torch.cuda.get_arch_list()
    say(f"torch arch list: {arches}")
    cap = torch.cuda.get_device_capability()
    say(f"device {torch.cuda.get_device_name()} capability {cap}")
    say(f"bf16 supported (native only): {torch.cuda.is_bf16_supported(including_emulation=False)}")
    ok = f"sm_{cap[0]}{cap[1]}" in arches
    say(f"torch carries SASS for this card: {ok}")
    import triton

    say(f"triton {triton.__version__}")
    import freetoken

    say(f"freetoken {freetoken.__version__ if hasattr(freetoken, '__version__') else '?'} at {freetoken.__file__}")
    from freetoken.version import __version__

    say(f"freetoken.version {__version__}")
    return ok


def compiler() -> None:
    stage("2. C compiler for Triton")
    try:
        from looma_ft_engine import provide_compiler

        say(f"CC = {provide_compiler() or '(none found)'}")
    except Exception as exc:
        say(f"!! provide_compiler failed: {exc!r}")


def kernel_cache() -> bool:
    stage("3. prebuilt kernel cache")
    from freetoken.kernel import aot, utils

    say(f"cache dir: {utils._kernel_cache_dir()}")
    good = True
    for spec in aot.default_kernel_specs():
        try:
            module = utils._load_prebuilt(spec.name)
            state = "ok" if module is not None else "MISSING"
        except Exception as exc:
            state = f"FAILED {type(exc).__name__}: {exc}"
        good &= state == "ok"
        say(f"  {spec.name}: {state}")
    return good


def numerics() -> bool:
    stage("4. matmul and activation kernels vs torch")
    import torch
    import torch.nn.functional as F

    from freetoken.kernel.triton import activation

    good = True
    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float16):
        name = str(dtype).split(".")[-1]
        try:
            a = torch.randn(512, 2048, device="cuda", dtype=dtype)
            b = torch.randn(2048, 1024, device="cuda", dtype=dtype)
            got = (a @ b).float()
            ref = a.float() @ b.float()
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                a @ b
            torch.cuda.synchronize()
            tflops = 20 * 2 * 512 * 2048 * 1024 / (time.perf_counter() - start) / 1e12
            say(f"  matmul {name}: rel err {err:.2e}, {tflops:.1f} TFLOPS")
            good &= err < 2e-2
        except Exception as exc:
            good = False
            say(f"  matmul {name}: FAILED {type(exc).__name__}: {exc}")
        x = torch.randn(64, 2 * 1536, device="cuda", dtype=dtype)
        gate, up = x.float().chunk(2, dim=-1)
        for kind, fn, ref in (
            ("silu", activation.silu_and_mul, F.silu(gate) * up),
            ("gelu_tanh", activation.gelu_tanh_and_mul, F.gelu(gate, approximate="tanh") * up),
        ):
            try:
                out = fn(x).float()
                err = ((out - ref).abs().max() / ref.abs().max()).item()
                say(f"  {kind}_and_mul {name}: rel err {err:.2e}")
                good &= err < 2e-2
            except Exception as exc:
                good = False
                say(f"  {kind}_and_mul {name}: FAILED {type(exc).__name__}: {str(exc)[:400]}")
    return good


def tests(dirs: list[str], timeout_min: float) -> None:
    stage(f"5. pytest {' '.join(dirs)}")
    if not os.path.exists("tests.tar.gz"):
        say("no tests.tar.gz among the inputs, skipped")
        return
    with tarfile.open("tests.tar.gz") as archive:
        archive.extractall("repo", filter="data")
    code, _ = run(sys.executable, "-m", "pytest", "--version", timeout=60)
    if code != 0:
        say("pytest is not installed in this env, skipped")
        return
    command = [sys.executable, "-m", "pytest", *[f"repo/tests/{d}" for d in dirs], "-q", "-rfE",
               "-p", "no:cacheprovider", "-m", "not slow and not needs_weights", "--tb=line"]
    code, text = run(*command, timeout=timeout_min * 60)
    lines = text.strip().splitlines()
    say("\n".join(lines[-150:]))
    say(f"pytest exit code {code}")


def health(port: int) -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=15) as answer:
            return json.loads(answer.read())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read())
        except Exception:
            return {"status": "?", "message": f"health returned {exc.code}"}
    except Exception as exc:
        return {"status": "?", "message": f"{type(exc).__name__}: {exc}"}


def ask(port: int, model: str, max_tokens: int, timeout: float) -> dict:
    payload = {"model": model, "max_tokens": max_tokens, "temperature": 0.0, "stream": False,
               "messages": [{"role": "user", "content": "Explain in two sentences what memory bandwidth is."}]}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            got = json.loads(answer.read())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "why": f"{exc.code}: {exc.read().decode(errors='replace')[:400]}"}
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    wall = time.perf_counter() - started
    tokens = int((got.get("usage") or {}).get("completion_tokens") or 0)
    text = ((got.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    return {"ok": True, "tokens": tokens, "wall_s": wall, "tps": tokens / wall if wall else 0.0, "text": text}


def serve(args) -> bool:
    stage("6. ft bench bw")
    code, text = run(*FT, "bench", "bw", timeout=1800)
    say(text.strip()[-3000:] or "(empty)")
    stage(f"6. ft serve --model {args.model} {' '.join(args.serve_arg)}")
    child = subprocess.Popen(
        [*FT, "serve", "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
         "--moe-strategy", "auto", *args.serve_arg],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump() -> None:
        for line in iter(child.stdout.readline, ""):
            print(f"  ft| {line.rstrip()}", flush=True)

    threading.Thread(target=pump, daemon=True).start()
    try:
        started, told = time.perf_counter(), 0.0
        while True:
            if child.poll() is not None:
                say(f"!! ft serve exited with {child.returncode}")
                return False
            state = health(args.port)
            if state.get("status") == "ok":
                break
            if state.get("status") == "error":
                say(f"!! server reported an error: {state.get('message')}")
                return False
            spent = time.perf_counter() - started
            if spent > args.wait_min * 60:
                say(f"!! not ready after {args.wait_min:.0f} min")
                return False
            if spent - told >= 60:
                told = spent
                progress = state.get("progress") or {}
                done, total = progress.get("done_bytes") or 0, progress.get("total_bytes") or 0
                where = f", {done / 1024**3:.1f}/{total / 1024**3:.1f} GB" if total else ""
                say(f"  ... {spent / 60:.0f} min: {state.get('status')} ({state.get('phase', '-')}{where}), card {card_memory()}")
            time.sleep(10)
        say(f"ready in {(time.perf_counter() - started) / 60:.1f} min, card {card_memory()}")
        for label in ("cold", "warm"):
            got = ask(args.port, args.model, args.max_tokens, timeout=args.wait_min * 60)
            if not got["ok"]:
                say(f"!! {label} request failed: {got['why']}")
                return False
            say(f"  {label}: {got['tokens']} tokens in {got['wall_s']:.1f} s = {got['tps']:.2f} tok/s")
            say(f"  answer: {got['text'][:300]!r}")
        code, text = run("free", "-g", timeout=30)
        say(text.strip())
        return True
    finally:
        child.terminate()
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            child.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--wait-min", type=float, default=120)
    parser.add_argument("--serve-arg", action="append", default=[],
                        help="extra ft serve argument, repeatable: --serve-arg=--dtype=float16")
    parser.add_argument("--test-dirs", default="kernels,attention,moe",
                        help="comma-separated tests/ subdirectories; empty skips stage 5")
    parser.add_argument("--test-timeout-min", type=float, default=25)
    parser.add_argument("--skip-serve", action="store_true")
    args = parser.parse_args()

    verdict = {}
    for name, step in (
        ("environment", environment),
        ("compiler", compiler),
        ("kernel_cache", kernel_cache),
        ("numerics", numerics),
    ):
        try:
            verdict[name] = step()
        except Exception:
            verdict[name] = False
            say(f"!! {name} crashed:\n{traceback.format_exc()}")
    dirs = [d for d in args.test_dirs.split(",") if d]
    if dirs:
        try:
            tests(dirs, args.test_timeout_min)
        except Exception:
            say(f"!! tests crashed:\n{traceback.format_exc()}")
    if not args.skip_serve:
        try:
            verdict["serve"] = serve(args)
        except Exception:
            verdict["serve"] = False
            say(f"!! serve crashed:\n{traceback.format_exc()}")
    stage("verdict")
    for name, ok in verdict.items():
        if ok is not None:
            say(f"  {name}: {'ok' if ok else 'FAILED'}")
    return 0 if all(v is not False for v in verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
