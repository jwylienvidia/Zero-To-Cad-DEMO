"""Subprocess sandbox for executing generated CadQuery code."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from zero_to_cad.config import SANDBOX_TIMEOUT_SEC

_RUNNER = r'''
import json
import sys
import traceback

import cadquery as cq

code = sys.stdin.read()
ns = {"cq": cq, "cadquery": cq}
try:
    exec(code, ns)
    result = ns.get("result")
    if result is None:
        raise ValueError("Code must define a variable named 'result'")
    cq.exporters.export(result, sys.argv[1], exportType="STEP")
    cq.exporters.export(result, sys.argv[2], exportType="STL")
    print(json.dumps({"ok": True}))
except Exception as e:
    print(json.dumps({"ok": False, "error": repr(e), "tb": traceback.format_exc()}))
    sys.exit(1)
'''


@dataclass
class ExecutionResult:
    ok: bool
    step_path: Path | None = None
    stl_path: Path | None = None
    error: str | None = None
    traceback: str | None = None
    work_dir: Path | None = None


def execute_cadquery(
    code: str,
    work_dir: Path | None = None,
    timeout: int = SANDBOX_TIMEOUT_SEC,
    python_executable: str | None = None,
) -> ExecutionResult:
    """
    Execute CadQuery code in an isolated subprocess.

    If work_dir is None, a temporary directory is created (caller should clean up
    unless using the returned work_dir for viewer loading).
    """
    if not code.strip():
        return ExecutionResult(ok=False, error="No code to execute")

    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="zero_to_cad_exec_"))
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    step_path = work_dir / "output.step"
    stl_path = work_dir / "output.stl"
    python = python_executable or sys.executable

    try:
        proc = subprocess.run(
            [python, "-c", _RUNNER, str(step_path), str(stl_path)],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(ok=False, error=f"Execution timed out after {timeout}s")

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    result_data: dict | None = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result_data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    if result_data and result_data.get("ok"):
        return ExecutionResult(
            ok=True,
            step_path=step_path if step_path.exists() else None,
            stl_path=stl_path if stl_path.exists() else None,
            work_dir=work_dir,
        )

    error = None
    tb = None
    if result_data:
        error = result_data.get("error")
        tb = result_data.get("tb")
    if not error:
        error = stderr or stdout or f"Process exited with code {proc.returncode}"

    return ExecutionResult(
        ok=False,
        error=error,
        traceback=tb,
    )
