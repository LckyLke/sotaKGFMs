#!/usr/bin/env python
"""Emit environment.json: the exact conditions the baseline ranks were produced under.

Records the ULTRA pin, the full pip freeze of the interpreter that ran it, the
accelerator and its driver/CUDA version, and the checkpoint hash.  Absent
hardware is recorded as absent, not omitted -- a missing GPU field would read as
an oversight rather than as the fact that there was no GPU.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def gpu_info(python):
    info = {"nvidia_smi_present": shutil.which("nvidia-smi") is not None}
    if info["nvidia_smi_present"]:
        query = run(["nvidia-smi",
                     "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"])
        info["nvidia_smi"] = query
    torch_probe = run([python, "-c", (
        "import json,torch;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'cuda_available': torch.cuda.is_available(),"
        "'cuda_version_torch_built_with': torch.version.cuda,"
        "'device_count': torch.cuda.device_count(),"
        "'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
        "}))")])
    if torch_probe:
        info.update(json.loads(torch_probe))
    info["nvcc"] = run(["nvcc", "--version"])
    info["cuda_home"] = os.environ.get("CUDA_HOME")
    if not info.get("cuda_available") and not info["nvidia_smi_present"]:
        info["note"] = (
            "No GPU present in this environment: no nvidia-smi, no nvcc, "
            "torch.cuda.is_available() is False. Ranks in ranks/ultra were "
            "produced on CPU. See baseline_report.md."
        )
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="interpreter that ran ULTRA")
    parser.add_argument("--ckpt", default=None, help="path to ultra_3g.pth")
    parser.add_argument("--out", default=os.path.join(WORKSPACE, "environment.json"))
    args = parser.parse_args(argv)

    pins = json.load(open(os.path.join(WORKSPACE, "repos", "PINS.json")))
    ckpt = args.ckpt or os.path.join(WORKSPACE, "repos", "ultra", "ckpts", "ultra_3g.pth")

    # uv-created virtualenvs have no pip module; importlib.metadata is always there
    freeze = run([args.python, "-m", "pip", "freeze"])
    if not freeze:
        freeze = run([args.python, "-c", (
            "import importlib.metadata as m;"
            "print('\\n'.join(sorted('%s==%s' % (d.metadata['Name'], d.version) "
            "for d in m.distributions())))")])
    freeze = freeze or ""
    env = {
        "generated_by": "scripts/environment.py",
        "ultra": {
            "sha": pins["repos"]["ultra"]["sha"],
            "url": pins["repos"]["ultra"]["url"],
            "patches": sorted(os.listdir(os.path.join(WORKSPACE, "patches", "ultra"))),
        },
        "checkpoint": {
            "name": os.path.basename(ckpt),
            "path": os.path.relpath(ckpt, WORKSPACE) if ckpt.startswith(WORKSPACE) else ckpt,
            "sha256": sha256(ckpt) if os.path.exists(ckpt) else None,
            "bytes": os.path.getsize(ckpt) if os.path.exists(ckpt) else None,
        },
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": run([args.python, "-c", "import sys;print(sys.version.split()[0])"]),
            "python_executable": args.python,
        },
        "accelerator": gpu_info(args.python),
        "pip_freeze": [line for line in freeze.splitlines() if line.strip()],
    }
    for name in ("ultra_3g", "ultra_4g", "ultra_50g"):
        candidate = os.path.join(os.path.dirname(ckpt), name + ".pth")
        if os.path.exists(candidate):
            env.setdefault("all_checkpoint_hashes", {})[name + ".pth"] = sha256(candidate)

    with open(args.out, "w") as handle:
        json.dump(env, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
