"""Task runner: background thread per run, streaming process output into an in-memory
ring buffer exposed by /api/logs."""

import json
import math
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from webui_app import configgen

PROJECT_DIR = Path(__file__).resolve().parent.parent
VENV_BIN = PROJECT_DIR / ".venv" / "bin"
MAX_LINES = 4000
MARK = "@@RESULT@@"


class Runner:
    def __init__(self):
        self.lock = threading.Lock()
        self.lines: list[str] = []
        self.base = 0
        self.proc: subprocess.Popen | None = None
        self.running = False
        self.stop_flag = False
        self.task = ""
        self.stage = ""
        self.derived: dict = {}
        self.last_probe: dict = {}
        self.started = 0.0
        self.progress: dict = {
            "step": 0,
            "total_steps": 0,
            "epoch": 0,
            "total_epochs": 0,
            "percent": 0.0,
            "loss": "-",
            "lr": "-",
            "speed": "-",
            "eta": "--:--:--",
        }
        self._last_step_time = 0.0

    # ----------------------------------------------------------------- log
    def log(self, text: str) -> None:
        self._parse_progress(text)
        with self.lock:
            self.lines.append(text if text.endswith("\n") else text + "\n")
            if len(self.lines) > MAX_LINES:
                drop = len(self.lines) - MAX_LINES
                del self.lines[:drop]
                self.base += drop

    def _parse_progress(self, text: str) -> None:
        total = self.progress.get("total_steps") or 0
        if not total:
            return

        # Pattern 1: StepLine format: step 42 loss=0.1234 grad_norm=... lr=1e-4
        m_step = re.search(r'\bstep\s+(?:\x1b\[[0-9;]*m)?(\d+)', text)
        if m_step:
            step = int(m_step.group(1))
            spc = self.derived.get("steps_per_epoch") or self.last_probe.get("steps_per_epoch") or 100
            total_epochs = self.progress.get("total_epochs") or 1
            curr_epoch = min(total_epochs, (step - 1) // spc + 1)
            pct = round((step / total) * 100, 1) if total > 0 else 0
            self.progress["step"] = step
            self.progress["epoch"] = curr_epoch
            self.progress["percent"] = pct

            m_loss = re.search(r'\bloss=([0-9\.]+)', text)
            if m_loss:
                self.progress["loss"] = f"{float(m_loss.group(1)):.4f}"

            m_lr = re.search(r'\blr=([0-9eE\.\-+]+)', text)
            if m_lr:
                self.progress["lr"] = m_lr.group(1)

            now = time.time()
            if self._last_step_time > 0 and getattr(self, "_last_step_num", 0) > 0 and step > self._last_step_num:
                d_steps = step - self._last_step_num
                dt = now - self._last_step_time
                if dt > 0:
                    self.progress["speed"] = f"{dt / d_steps:.2f} s/it"
            elif self._last_step_time > 0:
                dt = now - self._last_step_time
                if dt > 0:
                    self.progress["speed"] = f"{dt:.2f} s/it"
            self._last_step_time = now
            self._last_step_num = step

            if step > 0 and self.started > 0:
                elapsed = now - self.started
                sec_per_step = elapsed / step
                rem = int(max(0, (total - step) * sec_per_step))
                m, s = divmod(rem, 60)
                h, m = divmod(m, 60)
                self.progress["eta"] = f"{h:02d}:{m:02d}:{s:02d}"

        # Pattern 2: tqdm format: 50/400 [01:23<09:45, 1.67s/it, loss=0.1425, lr=...]
        m_tqdm = re.search(
            r'(?:(\d+)%\|.*\|\s*)?(\d+)/(\d+)\s*\[([^<]+)<([^,\]]+)(?:,\s*([\d\.]+(?:step/s|it/s|s/it)))?(?:,\s*loss=([\d\.]+))?(?:,\s*lr=([0-9eE\.\-+]+))?',
            text
        )
        if m_tqdm:
            step = int(m_tqdm.group(2))
            spc = self.derived.get("steps_per_epoch") or self.last_probe.get("steps_per_epoch") or 100
            total_epochs = self.progress.get("total_epochs") or (math.ceil(total / spc) if spc else 1)
            pct = float(m_tqdm.group(1)) if m_tqdm.group(1) else (round(step / total * 100, 1) if total else 0)
            self.progress["step"] = step
            self.progress["epoch"] = min(total_epochs, (step - 1) // spc + 1)
            self.progress["percent"] = pct
            self.progress["eta"] = m_tqdm.group(5).strip()
            if m_tqdm.group(6):
                self.progress["speed"] = m_tqdm.group(6).strip()
            if m_tqdm.group(7):
                self.progress["loss"] = f"{float(m_tqdm.group(7)):.4f}"
            if m_tqdm.group(8):
                self.progress["lr"] = m_tqdm.group(8).strip()

    def read(self, since: int) -> dict:
        with self.lock:
            start = max(since, self.base)
            chunk = self.lines[start - self.base:]
            return {"lines": chunk, "next": self.base + len(self.lines), "reset": since < self.base}

    def clear(self) -> None:
        with self.lock:
            self.base += len(self.lines)
            self.lines.clear()

    def status(self) -> dict:
        return {
            "running": self.running,
            "task": self.task,
            "stage": self.stage,
            "elapsed": int(time.time() - self.started) if self.running else 0,
            "derived": self.derived,
            "last_probe": self.last_probe,
            "progress": self.progress,
        }

    # ------------------------------------------------------------- process
    def _env(self) -> dict:
        env = os.environ.copy()
        env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(PROJECT_DIR / ".venv")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(PROJECT_DIR) + os.pathsep + env.get("PYTHONPATH", "")

        # Auto-inject nvidia libraries (cuDNN, cuBLAS, etc.) into LD_LIBRARY_PATH
        # Fixes: RuntimeError: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
        ld_dirs: list[str] = []
        venv_dir = PROJECT_DIR / ".venv"
        for sp in venv_dir.glob("lib/python*/site-packages/nvidia"):
            for sub in ("cudnn/lib", "cublas/lib", "cuda_runtime/lib", "cusparse/lib"):
                lib_path = sp / sub
                if lib_path.is_dir():
                    ld_dirs.append(str(lib_path))

        if ld_dirs:
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            all_ld = ld_dirs + ([existing_ld] if existing_ld else [])
            env["LD_LIBRARY_PATH"] = ":".join(all_ld)

        return env

    def _run(self, cmd: list[str], capture: bool = False) -> tuple[int, str]:
        """Run one command, streaming to the log; returns (code, captured text)."""
        self.log(">>> " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            env=self._env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            preexec_fn=os.setsid,
        )
        self.proc = proc
        kept = []
        for line in iter(proc.stdout.readline, ""):
            if capture and line.startswith(MARK):
                kept.append(line)
                continue
            self.log(line.rstrip("\r\n").replace("\r", "\n"))
        proc.stdout.close()
        code = proc.wait()
        self.proc = None
        return code, "".join(kept)

    def stop(self) -> None:
        self.stop_flag = True
        proc = self.proc
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                proc.terminate()
            self.log(">>> [system] 已请求终止当前任务")

    def _start(self, task: str, target, *args) -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.stop_flag = False
            self.task = task
            self.stage = ""
            self.started = time.time()

        def wrapper():
            try:
                target(*args)
            except Exception as err:  # surfaced in the console
                self.log(f">>> [error] {type(err).__name__}: {err}")
            finally:
                self.running = False
                self.stage = ""
                self.log(f">>> [{task}] 结束")

        threading.Thread(target=wrapper, daemon=True).start()
        return True

    # --------------------------------------------------------------- tasks
    def doctor(self) -> bool:
        return self._start("Doctor", lambda: self._run([str(VENV_BIN / "kohakumoss-doctor")]))

    def probe(self, params: dict) -> bool:
        return self._start("Probe", self._probe_job, params)

    def train(self, params: dict) -> bool:
        return self._start("Training", self._train_job, params)

    def _write(self, params: dict, steps_per_epoch: int) -> tuple[Path, Path, dict]:
        text, derived = configgen.build_config(params, steps_per_epoch)
        safe_fn = getattr(configgen, "safe_run_name", None) or (lambda s: "".join(c for c in str(s or "") if c.isalnum() or c in "-_.").strip() or "run")
        run_name = safe_fn(params.get("run_name", "run"))
        out_root = Path(configgen.to_linux_path(params.get("output_dir", "out") or "out"))
        run_dir = out_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = run_dir / "config.py"
        cfg.write_text(text, encoding="utf-8")
        data_cfg = run_dir / "data_config.py"
        data_cfg.write_text(text + configgen.data_overrides(params), encoding="utf-8")
        (run_dir / "ui_params.json").write_text(
            json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return cfg, data_cfg, derived

    def _do_probe(self, params: dict, data_cfg: Path) -> dict:
        self.stage = "扫描数据集"
        code, out = self._run([str(VENV_BIN / "python"), "-m", "webui_app.probe", str(data_cfg)], True)
        if code != 0 or not out:
            raise RuntimeError("数据集探测失败，请查看上方日志（路径/目录结构是否正确）")
        return json.loads(out.strip().splitlines()[-1][len(MARK):])

    def _probe_job(self, params: dict) -> None:
        _, data_cfg, _ = self._write(params, 100)
        result = self._do_probe(params, data_cfg)
        self.last_probe = result
        self.log(">>> [probe] " + json.dumps(result["plan"], ensure_ascii=False))
        self.log(f">>> [probe] 每 epoch 约 {result['steps_per_epoch']} 步，目标尺寸 {result['distinct_targets']} 种")

    def _train_job(self, params: dict) -> None:
        p = configgen.merged(params)
        _, data_cfg, _ = self._write(params, 100)
        result = self._do_probe(params, data_cfg)
        self.last_probe = result
        spc = result["steps_per_epoch"]
        self.log(f">>> [plan] 每 epoch {spc} 步 (批次 {result['batches']}, 梯度累加 {result['grad_acc']})")
        cfg, data_cfg, derived = self._write(params, spc)
        self.derived = derived
        self.log(
            f">>> [plan] 总步数 {derived['steps']}，每 {derived['export_every']} 步导出 LoRA，"
            f"每 {derived['save_every']} 步保存续训检查点，有效批量 {derived['effective_batch']}"
        )
        total_steps = derived.get("steps", 0)
        total_epochs = math.ceil(total_steps / spc) if spc else 1
        self.progress = {
            "step": 0,
            "total_steps": total_steps,
            "epoch": 1,
            "total_epochs": total_epochs,
            "percent": 0.0,
            "loss": "-",
            "lr": "-",
            "speed": "-",
            "eta": "--:--:--",
        }
        self._last_step_time = 0.0

        if self.stop_flag:
            return
        kogine = str(VENV_BIN / "kogine")
        if p["text_cache"] == "offline" and not p["skip_cache_step"]:
            self.stage = "缓存文本嵌入"
            if self._run([kogine, "run", "scripts/data/cache_text.py", "--config", str(data_cfg)])[0] or self.stop_flag:
                raise RuntimeError("文本缓存失败/被终止")
        if p["latent_cache"] == "offline" and not p["skip_cache_step"]:
            self.stage = "缓存 Latent"
            if self._run([kogine, "run", "scripts/data/cache_latents.py", "--config", str(data_cfg)])[0] or self.stop_flag:
                raise RuntimeError("Latent 缓存失败/被终止")
        self.stage = "训练中"
        code, _ = self._run([kogine, "run", "scripts/train/moss.py", "--config", str(cfg)])
        if code == 0 and not self.stop_flag and total_steps > 0:
            self.progress["step"] = self.progress["total_steps"]
            self.progress["epoch"] = self.progress["total_epochs"]
            self.progress["percent"] = 100.0
            self.progress["eta"] = "已完成"


RUNNER = Runner()
