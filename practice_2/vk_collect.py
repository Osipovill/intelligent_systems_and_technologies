"""Legacy-обёртка для старого снимка четырёх участников.

Основная реализация сбора находится в :mod:`collect_group`. Этот файл сохраняет
прежнюю команду запуска и путь старого набора, но не дублирует сетевой конвейер.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from collect_group import CSV_COLUMNS, build_csv, empty_state, run_collection
from config import BASE_DIR, LEGACY_MEMBERS


def self_check():
    """Проверяет экспорт legacy-схемы без обращения к VK API."""

    state = empty_state()
    state["member_friends"] = {"198182178": [10]}
    state["direct_friends"] = {"10": [20]}
    state["profiles"] = {"10": {"id": 10}, "20": {"id": 20}}
    state["list_statuses"] = {"198182178": "loaded", "10": "loaded"}
    output_path = Path("/tmp/vk_collect_self_check.csv")
    row_count, direct_ids, second_ids = build_csv(
        state,
        output_path,
        LEGACY_MEMBERS,
    )
    with output_path.open(encoding="utf-8", newline="") as file:
        assert next(csv.reader(file)) == CSV_COLUMNS
    assert row_count == 2
    assert direct_ids == {10}
    assert second_ids == {20}
    output_path.unlink()
    print("Legacy self-check passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.self_check:
        self_check()
        return
    run_collection(
        BASE_DIR / "data",
        cache_from=BASE_DIR / "data" / "collect_state.json",
        env_path=BASE_DIR / ".env",
        member_refs=[str(user_id) for user_id in LEGACY_MEMBERS],
    )


if __name__ == "__main__":
    main()
