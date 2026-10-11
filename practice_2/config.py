"""Единая конфигурация сбора данных и анализа графа VK."""

from __future__ import annotations

import hashlib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
API_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"
PROFILE_FIELDS = "screen_name,is_closed,deactivated"
PROFILE_BATCH_SIZE = 500
FRIENDS_PAGE_SIZE = 5000
REQUEST_INTERVAL = 0.5
TOKEN_COOLDOWN = 60

CENTRALITY_K = 128
CENTRALITY_K_VALUES = (64, 128)
CENTRALITY_SEEDS = (42, 43)
CENTRALITY_BACKEND = "auto"

OVERVIEW_SECOND_LIMIT = 305
MAX_OVERVIEW_NODES = 650
SNAPSHOT_VERSION = "2.1"

GROUP_IDS_PATH = BASE_DIR / "data" / "group_ids.csv"
GROUP_DATA_DIR = BASE_DIR / "data" / "group"

# Эти участники относятся только к сохранённому legacy-сценарию.
LEGACY_MEMBERS = {
    198182178: "flashik12",
    473310225: "0pupok00",
    424935969: "ryosuk3",
    465105120: "medvediano",
}


def sha256_file(path: Path) -> str:
    """Возвращает SHA-256 файла для проверки неизменности снимка данных."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
