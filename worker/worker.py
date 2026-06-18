#!/usr/bin/env python3
"""
AI Studio GPU Worker.

Ce worker expose un serveur HTTP local. Le serveur Hugging Face Space l'appelle
via l'URL ngrok enregistrée dans l'interface. Les requêtes du Space vers ngrok
incluent le header `ngrok-skip-browser-warning: true`.

Installation minimale:
  pip install psutil pynvml

Pour l'entraînement Hugging Face:
  pip install torch transformers datasets peft accelerate evaluate scikit-learn

Lancement exemple:
  python worker.py --server-url https://ton-space.hf.space --user-token TON_TOKEN --worker-name "Colab T4" --port 8765
"""

import argparse
import json
import signal
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import psutil
except Exception:
    psutil = None

try:
    import pynvml
except Exception:
    pynvml = None

STATE = {
    "server_url": "",
    "user_token": "",
    "worker_name": "",
    "ngrok_url": "",
    "current_job": None,
    "stop_event": None,
    "lock": threading.Lock(),
    "started_at": time.time()
}


def log(message):
    print(f"[worker] {message}", flush=True)


def post_json(path, payload, token=None, timeout=20):
    url = f"{STATE['server_url'].rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, data=body, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} -> HTTP {error.code}: {raw}") from error
    except URLError as error:
        raise RuntimeError(f"POST {path} -> {error}") from error


def send_job_log(job_id, level, message, progress=None):
    try:
        post_json(
            f"/api/jobs/{job_id}/logs",
            {
                "level": level,
                "message": message,
                "progress": progress
            },
            token=STATE["user_token"],
            timeout=15
        )
    except Exception as exc:
        log(f"Impossible d'envoyer le log: {exc}")


def send_job_status(job_id, status, progress=None, error=None):
    try:
        post_json(
            f"/api/jobs/{job_id}/status",
            {
                "status": status,
                "progress": progress,
                "error": error
            },
            token=STATE["user_token"],
            timeout=20
        )
    except Exception as exc:
        log(f"Impossible d'envoyer le statut: {exc}")


def decode_nvml_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def get_gpu_metrics():
    metrics = {
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_index": 0,
        "vram_total": 0,
        "vram_used": 0,
        "vram_free": 0,
        "gpu_util": 0,
        "gpu_temperature": None
    }

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            metrics["gpu_count"] = count

            if count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = decode_nvml_value(pynvml.nvmlDeviceGetName(handle))
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

                metrics.update({
                    "gpu_name": name,
                    "gpu_index": 0,
                    "vram_total": int(memory.total),
                    "vram_used": int(memory.used),
                    "vram_free": int(memory.free),
                    "gpu_util": int(utilization.gpu),
                    "gpu_temperature": int(pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU
                    ))
                })

            pynvml.nvmlShutdown()
            return metrics
        except Exception as exc:
            metrics["gpu_error"] = f"pynvml: {exc}"

    try:
        import subprocess

        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        lines = [line for line in result.stdout.splitlines() if line.strip()]

        if lines:
            parts = [part.strip() for part in lines[0].split(",")]
            if len(parts) >= 6:
                metrics.update({
                    "gpu_count": len(lines),
                    "gpu_name": parts[0],
                    "gpu_index": 0,
                    "vram_total": int(float(parts[1]) * 1024 * 1024),
                    "vram_used": int(float(parts[2]) * 1024 * 1024),
                    "vram_free": int(float(parts[3]) * 1024 * 1024),
                    "gpu_util": int(float(parts[4])),
                    "gpu_temperature": int(float(parts[5]))
                })
    except Exception as exc:
        if "gpu_error" not in metrics:
            metrics["gpu_error"] = f"nvidia-smi: {exc}"

    return metrics


def get_metrics():
    metrics = {
        "worker_url": STATE["ngrok_url"],
        "worker_name": STATE["worker_name"],
        "uptime_seconds": int(time.time() - STATE["started_at"]),
        "current_job": STATE["current_job"],
        "python_version": sys.version.split()[0]
    }

    if psutil is not None:
        try:
            cpu = psutil.cpu_percent(interval=0.2)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            metrics.update({
                "cpu_percent": cpu,
                "cpu_count": psutil.cpu_count(logical=True),
                "ram_total": int(memory.total),
                "ram_used": int(memory.used),
                "ram_free": int(memory.free),
                "ram_percent": memory.percent,
                "disk_total": int(disk.total),
                "disk_used": int(disk.used),
                "disk_free": int(disk.free),
                "disk_percent": disk.percent
            })
        except Exception as exc:
            metrics["system_error"] = f"psutil: {exc}"
    else:
        metrics["system_error"] = "psutil non installé"

    metrics.update(get_gpu_metrics())
    return metrics


def register_worker():
    try:
        with STATE["lock"]:
            current_job = STATE["current_job"]

        post_json(
            "/api/workers/register",
            {
                "userToken": STATE["user_token"],
                "workerName": STATE["worker_name"],
                "ngrokUrl": STATE["ngrok_url"],
                "status": "busy" if current_job else "online",
                "metrics": get_metrics()
            },
            timeout=15
        )
        log("Worker enregistré auprès du serveur.")
    except Exception as exc:
        log(f"Échec de l'enregistrement: {exc}")


def heartbeat_loop(interval):
    while True:
        time.sleep(interval)
        register_worker()


def run_job_thread(job, stop_event):
    job_id = job.get("jobId")
    log(f"Démarrage du job {job_id}")
    send_job_status(job_id, "running", progress=0)
    send_job_log(job_id, "info", "Démarrage de l'entraînement sur le worker GPU.", progress=0)

    try:
        from train_hf import run_training

        result = run_training(
            job=job,
            log_callback=lambda level, message, progress=None: send_job_log(job_id, level, message, progress),
            stop_event=stop_event
        )

        if stop_event.is_set():
            send_job_status(job_id, "cancelled", progress=100)
            send_job_log(job_id, "warning", "Job arrêté par l'utilisateur.", progress=100)
        else:
            send_job_status(job_id, "completed", progress=100)
            send_job_log(
                job_id,
                "success",
                f"Entraînement terminé. Sortie: {result.get('output_dir', 'inconnue')}",
                progress=100
            )
    except KeyboardInterrupt:
        send_job_status(job_id, "cancelled", progress=100)
        send_job_log(job_id, "warning", "Job annulé.", progress=100)
    except Exception as exc:
        log(traceback.format_exc())
        send_job_status(job_id, "failed", error=str(exc))
        send_job_log(job_id, "error", f"Erreur pendant l'entraînement: {exc}")
    finally:
        with STATE["lock"]:
            STATE["current_job"] = None
            STATE["stop_event"] = None
        log(f"Job {job_id} terminé.")


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "AIStudioWorker/1.0"

    def log_message(self, format, *args):
        log(f"HTTP {args[0] if args else ''}")

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, ngrok-skip-browser-warning")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if not length:
            return {}

        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, ngrok-skip-browser-warning")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            with STATE["lock"]:
                current_job = STATE["current_job"]
                stop_requested = bool(STATE["stop_event"].is_set() if STATE["stop_event"] else False)

            self.send_json(200, {
                "ok": True,
                "status": "busy" if current_job else "idle",
                "current_job": current_job,
                "stop_requested": stop_requested,
                "ngrok_url": STATE["ngrok_url"],
                "worker_name": STATE["worker_name"],
                "metrics": get_metrics()
            })
            return

        if self.path == "/metrics":
            self.send_json(200, get_metrics())
            return

        self.send_json(404, {"error": "Endpoint inconnu."})

    def do_POST(self):
        try:
            payload = self.read_json()

            if self.path == "/start-job":
                with STATE["lock"]:
                    if STATE["current_job"]:
                        self.send_json(409, {
                            "ok": False,
                            "error": "Un job est déjà en cours sur ce worker."
                        })
                        return

                    STATE["current_job"] = payload.get("jobId")
                    STATE["stop_event"] = threading.Event()
                    stop_event = STATE["stop_event"]

                thread = threading.Thread(
                    target=run_job_thread,
                    args=(payload, stop_event),
                    daemon=True
                )
                thread.start()
                self.send_json(200, {"ok": True, "jobId": payload.get("jobId")})
                return

            if self.path == "/stop-job":
                job_id = payload.get("jobId")
                with STATE["lock"]:
                    current_job = STATE["current_job"]
                    if current_job and current_job == job_id and STATE["stop_event"]:
                        STATE["stop_event"].set()
                        self.send_json(200, {"ok": True, "jobId": job_id, "stop_requested": True})
                    else:
                        self.send_json(404, {"ok": False, "error": "Aucun job correspondant n'est en cours."})
                return

            self.send_json(404, {"error": "Endpoint inconnu."})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})


def parse_args():
    parser = argparse.ArgumentParser(description="AI Studio GPU Worker")
    parser.add_argument("--server-url", required=True, help="URL du serveur AI Studio, par exemple https://ton-space.hf.space")
    parser.add_argument("--user-token", required=True, help="Token utilisateur pour s'authentifier auprès du serveur")
    parser.add_argument("--worker-name", default="GPU Worker", help="Nom de la machine worker")
    parser.add_argument("--ngrok-url", default="", help="URL publique ngrok, utile pour l'affichage et le diagnostic")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--heartbeat-interval", type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    STATE["server_url"] = args.server_url.rstrip("/")
    STATE["user_token"] = args.user_token
    STATE["worker_name"] = args.worker_name
    STATE["ngrok_url"] = args.ngrok_url

    def shutdown(signum, frame):
        log("Signal reçu, arrêt du worker...")
        with STATE["lock"]:
            if STATE["stop_event"]:
                STATE["stop_event"].set()
        server.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    register_worker()

    heartbeat = threading.Thread(
        target=heartbeat_loop,
        args=(args.heartbeat_interval,),
        daemon=True
    )
    heartbeat.start()

    global server
    server = ThreadingHTTPServer((args.host, args.port), WorkerHandler)
    log(f"Worker HTTP démarré sur {args.host}:{args.port}")
    log(f"Worker name: {STATE['worker_name']}")
    log("Le serveur Space doit utiliser l'URL ngrok enregistrée dans l'interface.")
    server.serve_forever()


if __name__ == "__main__":
    main()