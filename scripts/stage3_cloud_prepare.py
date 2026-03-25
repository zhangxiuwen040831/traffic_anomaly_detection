#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import paramiko

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.stage3.config import load_stage3_config
from src.stage3.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the weak cloud host for future Stage 3 GPU training")
    parser.add_argument("--config", default="config/stage3/base.yaml")
    parser.add_argument("--host", default="qhdlink.lanyun.net")
    parser.add_argument("--port", type=int, default=44410)
    parser.add_argument("--username", default="root")
    parser.add_argument("--password", required=True)
    parser.add_argument("--remote-project-root", default="/root/lanyun-fs/traffic_anomaly_detection_stage3")
    return parser.parse_args()


def build_source_bundle(project_root: Path) -> bytes:
    include_paths = [
        "src",
        "scripts",
        "config",
        "requirements.txt",
        "README.md",
        "complete_report.md",
    ]
    exclude_prefixes = {
        ".venv",
        "__pycache__",
        "outputs",
        "outputs_stage3",
        "data/raw",
        "data/processed",
        "data/stage3",
        "config/data",
    }

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in include_paths:
            source = project_root / relative
            if not source.exists():
                continue
            if source.is_file():
                archive.add(source, arcname=source.relative_to(project_root))
                continue
            for path in source.rglob("*"):
                rel = path.relative_to(project_root)
                rel_str = str(rel).replace("\\", "/")
                if any(rel_str == prefix or rel_str.startswith(f"{prefix}/") for prefix in exclude_prefixes):
                    continue
                if "__pycache__" in path.parts:
                    continue
                archive.add(path, arcname=rel)
    return buffer.getvalue()


def run_remote(client: paramiko.SSHClient, command: str, timeout: int = 120) -> dict[str, object]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return {
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout.read().decode("utf-8", errors="replace"),
        "stderr": stderr.read().decode("utf-8", errors="replace"),
    }


def main() -> int:
    args = parse_args()
    config = load_stage3_config(args.config)
    local_output_root = ensure_dir(PROJECT_ROOT / config["outputs"]["root"] / "cloud_prepare")
    bundle_path = local_output_root / "stage3_source_bundle.tar.gz"
    report_path = local_output_root / "cloud_prepare_report.md"
    summary_path = local_output_root / "cloud_prepare_summary.json"
    local_probe_copy = local_output_root / "remote_environment_probe.txt"

    bundle_bytes = build_source_bundle(PROJECT_ROOT)
    bundle_path.write_bytes(bundle_bytes)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        timeout=30,
    )

    sftp = client.open_sftp()
    remote_bundle = f"{args.remote_project_root}/stage3_source_bundle.tar.gz"
    commands = []
    try:
        commands.append(run_remote(client, f"mkdir -p {args.remote_project_root}"))
        with sftp.file(remote_bundle, "wb") as remote_handle:
            remote_handle.write(bundle_bytes)

        commands.append(run_remote(client, f"cd {args.remote_project_root} && tar -xzf stage3_source_bundle.tar.gz && rm -f stage3_source_bundle.tar.gz"))
        commands.append(run_remote(client, f"cd {args.remote_project_root} && chmod +x scripts/cloud/*.sh"))
        commands.append(run_remote(client, f"cd {args.remote_project_root} && bash scripts/cloud/stage3_prepare_layout.sh {args.remote_project_root}"))
        commands.append(run_remote(client, f"cd {args.remote_project_root} && bash scripts/cloud/stage3_env_probe.sh outputs_stage3/cloud_prepare/reports"))
        commands.append(run_remote(client, f"cd {args.remote_project_root} && find outputs_stage3 -maxdepth 3 -type d | sort"))

        remote_probe = f"{args.remote_project_root}/outputs_stage3/cloud_prepare/reports/environment_probe.txt"
        sftp.get(remote_probe, str(local_probe_copy))
    finally:
        sftp.close()
        client.close()

    payload = {
        "remote_host": args.host,
        "remote_port": args.port,
        "remote_project_root": args.remote_project_root,
        "bundle_size_bytes": len(bundle_bytes),
        "bundle_path": str(bundle_path),
        "local_probe_copy": str(local_probe_copy),
        "commands": commands,
    }
    save_json(summary_path, payload)

    report_lines = [
        "# Stage 3 Cloud Prepare Report",
        "",
        "## Remote Target",
        f"- Host: {args.host}:{args.port}",
        f"- Remote project root: {args.remote_project_root}",
        f"- Uploaded source bundle: {bundle_path.name} ({len(bundle_bytes)} bytes)",
        "",
        "## Executed Steps",
    ]
    for item in commands:
        report_lines.append(f"- Exit {item['exit_code']}: `{item['command']}`")
    report_lines.extend(
        [
            "",
            "## Environment Probe Copy",
            f"- Local copy: {local_probe_copy}",
            "",
            "## Future GPU Launchers",
            f"- `bash scripts/cloud/stage3_download_datasets.sh {args.remote_project_root} unsw_nb15`",
            f"- `bash scripts/cloud/stage3_download_datasets.sh {args.remote_project_root} cic_ids2017`",
            f"- `LAUNCH_MODE=tmux MODEL_NAME=enhanced_mlp_ae DATASET_NAME=unsw_nb15 bash scripts/cloud/stage3_gpu_train.sh {args.remote_project_root}`",
            f"- `LAUNCH_MODE=nohup MODEL_NAME=hybrid_ae DATASET_NAME=cic_ids2017 bash scripts/cloud/stage3_gpu_train.sh {args.remote_project_root}`",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Cloud prepare complete. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
