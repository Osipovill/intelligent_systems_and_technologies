"""Безопасно подготавливает фиксированную локальную среду Ollama для практикума"""

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request

CONTAINER = "practice3-ollama"
IMAGE = "ollama/ollama:0.30.7"
NETWORK = "practice3-private"
VOLUME = "practice3-ollama-models"
MEMORY = 6 * 1024**3


def docker(args, timeout=30):
    """Выполняет Docker CLI с фиксированными аргументами без shell-интерпретации"""
    if not shutil.which("docker"):
        raise RuntimeError(
            "Docker CLI is unavailable; start the bootstrap Jupyter service"
        )
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Docker command failed")
    return result.stdout.strip()


def api_ready(url="http://practice3-ollama:11434", timeout=90):
    """Ограниченно ожидает готовность API и сообщает понятную причину сбоя"""
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/version", timeout=3) as response:
                version = json.load(response).get("version")
                if version != "0.30.7":
                    raise RuntimeError(
                        f"Unsupported Ollama version {version}; expected 0.30.7"
                    )
                return
        except RuntimeError:
            raise
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = str(error)
            time.sleep(2)
    raise RuntimeError(f"Ollama API readiness timeout: {last_error}")


def bootstrap():
    """Создаёт или запускает только заранее выбранные ресурсы практикума"""
    if not shutil.which("docker"):
        raise RuntimeError(
            "Docker CLI is unavailable; rebuild the bootstrap Jupyter image"
        )
    if not __import__("os").path.exists("/var/run/docker.sock"):
        raise RuntimeError(
            "Docker socket is unavailable; enable the explicitly authorized socket mount"
        )
    (
        docker(["network", "inspect", NETWORK])
        if _exists("network", NETWORK)
        else docker(["network", "create", NETWORK])
    )
    (
        docker(["volume", "inspect", VOLUME])
        if _exists("volume", VOLUME)
        else docker(["volume", "create", VOLUME])
    )
    if _exists("container", CONTAINER):
        config = json.loads(docker(["inspect", CONTAINER]))[0]
        image = config["Config"]["Image"]
        volumes = config["Mounts"]
        gpu = config["HostConfig"].get("DeviceRequests", [])
        host = config["HostConfig"]
        compatible = (
            image in (IMAGE, "ollama/ollama:latest")
            and config["HostConfig"]["NetworkMode"] == NETWORK
            and any(
                item.get("Name") == VOLUME
                and item.get("Destination") == "/root/.ollama"
                for item in volumes
            )
            and any("gpu" in request.get("Capabilities", [[]])[0] for request in gpu)
        )
        if not compatible:
            raise RuntimeError(
                "Existing Ollama container is incompatible; inspect it manually, it was not modified"
            )
        status = config["State"]["Status"]
        if status == "running" and (
            host.get("Memory") != MEMORY or host.get("MemorySwap") != MEMORY
        ):
            raise RuntimeError(
                "Running Ollama container lacks the required 6 GiB memory and swap caps; stop your Ollama container, then rerun this notebook"
            )
        if status != "running":
            if host.get("Memory") != MEMORY or host.get("MemorySwap") != MEMORY:
                docker(["update", "--memory=6g", "--memory-swap=6g", CONTAINER])
            docker(["start", CONTAINER])
    else:
        docker(
            [
                "run",
                "-d",
                "--name",
                CONTAINER,
                "--network",
                NETWORK,
                "--gpus",
                "all",
                "--memory=6g",
                "--memory-swap=6g",
                "-v",
                f"{VOLUME}:/root/.ollama",
                "-e",
                "OLLAMA_FLASH_ATTENTION=0",
                "-e",
                "OLLAMA_GPU_OVERHEAD=536870912",
                "-e",
                "OLLAMA_NUM_PARALLEL=1",
                "-e",
                "OLLAMA_MAX_LOADED_MODELS=1",
                IMAGE,
            ]
        )
    api_ready()


def _exists(kind, name):
    """Проверяет наличие ресурса без воздействия на чужие контейнеры"""
    result = subprocess.run(
        ["docker", kind, "inspect", name], capture_output=True, text=True
    )
    if result.returncode and "No such" not in result.stderr:
        raise RuntimeError(result.stderr.strip())
    return result.returncode == 0
