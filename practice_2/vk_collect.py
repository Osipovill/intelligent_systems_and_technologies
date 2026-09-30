"""Собирает два круга друзей VK с чекпоинтом: клиент выполняет запросы, конвейер сохраняет списки и профили, сборщик записывает единый CSV"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests


# Участники задают канонический порядок родителей в итоговом наборе данных
GROUP_MEMBERS = {
    198182178: "flashik12",
    473310225: "0pupok00",
    424935969: "ryosuk3",
    465105120: "medvediano",
}
API_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"
PROFILE_FIELDS = "screen_name,is_closed,deactivated"
PROFILE_BATCH_SIZE = 500
REQUEST_INTERVAL = 0.35
TOKEN_COOLDOWN = 60
CSV_COLUMNS = [
    "user_id", "friend_id", "observation_level", "user_screen_name",
    "user_is_group_member", "friend_first_name", "friend_last_name",
    "friend_screen_name", "friend_is_closed", "friend_deactivated",
    "friend_profile_status", "friend_discovery_level", "friend_parent_ids",
    "friend_parent_screen_name", "friend_via_friend_ids",
    "friend_friends_status", "friend_friends_count",
]


def configure_logging():
    """Настраивает единый поток журналирования для длительного запуска"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_tokens():
    """Читает сервисные токены из окружения, не сохраняя их в файлах"""
    tokens = [token.strip() for token in os.environ.get("VK_SERVICE_TOKENS", "").split(",") if token.strip()]
    if not tokens:
        token = os.environ.get("VK_SERVICE_TOKEN", "").strip()
        if token:
            tokens = [token]
    if not tokens:
        raise RuntimeError("VK_SERVICE_TOKENS or VK_SERVICE_TOKEN must be set")
    return tokens


class VkClient:
    """Выполняет VK API запросы с общим темпом и перерывами для токенов"""

    def __init__(self, tokens):
        """Инициализирует ротацию переданными сервисными токенами"""
        self.tokens = tokens
        self.active_index = 0
        self.cooldown_until = {token: 0.0 for token in tokens}
        self.last_request_at = 0.0

    def _select_token(self):
        """Выбирает доступный токен или ждёт до ближайшего окончания перерыва"""
        while True:
            now = time.monotonic()
            for offset in range(len(self.tokens)):
                index = (self.active_index + offset) % len(self.tokens)
                token = self.tokens[index]
                if self.cooldown_until[token] <= now:
                    self.active_index = index
                    return token
            next_ready = min(self.cooldown_until.values())
            wait_seconds = max(0.0, next_ready - now)
            logging.warning("All tokens are cooling down; waiting %.1f seconds", wait_seconds)
            time.sleep(wait_seconds)

    def _wait_for_rate_limit(self):
        """Выдерживает минимальный интервал между запросами независимо от токена"""
        wait_seconds = REQUEST_INTERVAL - (time.monotonic() - self.last_request_at)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def call(self, method, params):
        """Вызывает метод VK, ротируя токен при ошибках ограничения частоты"""
        while True:
            token = self._select_token()
            self._wait_for_rate_limit()
            request_params = {**params, "access_token": token, "v": API_VERSION}
            try:
                response = requests.get(API_URL + method, params=request_params, timeout=30)
                self.last_request_at = time.monotonic()
                payload = response.json()
            except (requests.RequestException, ValueError) as error:
                self.last_request_at = time.monotonic()
                return {"__error__": {"error_code": -1, "error_msg": str(error)}}
            error = payload.get("error")
            if not error:
                return payload.get("response", {})
            code = error.get("error_code")
            if code == 30:
                return {"__profile_closed__": True}
            if code not in (6, 9):
                return {"__error__": error}
            self.cooldown_until[token] = time.monotonic() + TOKEN_COOLDOWN
            self.active_index = (self.active_index + 1) % len(self.tokens)
            logging.warning("Token entered cooldown after VK error %s", code)


def empty_state():
    """Создаёт минимальное состояние, которое сериализуется в JSON"""
    return {"member_friends": {}, "direct_friends": {}, "profiles": {}, "list_statuses": {}}


def load_state(path):
    """Загружает сохранённый прогресс либо создаёт пустое состояние"""
    if not path.exists():
        return empty_state()
    with path.open(encoding="utf-8") as file:
        state = json.load(file)
    defaults = empty_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    logging.info("Resuming collection from checkpoint %s", path)
    return state


def save_state(path, state):
    """Атомарно сохраняет прогресс, чтобы повторный запуск не терял собранное"""
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False)
    temporary_path.replace(path)


def profile_status(profile):
    """Возвращает явный статус доступности анкеты из чекпоинта"""
    return profile.get("_profile_status", "loaded" if profile else "not_loaded")


def collect_profiles(client, state, user_ids, state_path, label):
    """Запрашивает отсутствующие профили строго пачками не более 500 идентификаторов"""
    missing_ids = [user_id for user_id in sorted(set(user_ids)) if str(user_id) not in state["profiles"]]
    for batch_number, start in enumerate(range(0, len(missing_ids), PROFILE_BATCH_SIZE), start=1):
        batch = missing_ids[start:start + PROFILE_BATCH_SIZE]
        response = client.call("users.get", {"user_ids": ",".join(map(str, batch)), "fields": PROFILE_FIELDS})
        if "__error__" in response:
            error = response["__error__"]
            logging.error("Profile batch %s failed: %s", batch_number, error.get("error_msg", "unknown error"))
            for user_id in batch:
                state["profiles"][str(user_id)] = {"id": user_id, "_profile_status": f"error_{error.get('error_code', -1)}"}
        else:
            received = {profile["id"]: profile for profile in response if isinstance(profile, dict) and "id" in profile}
            for user_id in batch:
                state["profiles"][str(user_id)] = received.get(
                    user_id, {"id": user_id, "_profile_status": "not_returned"}
                )
        save_state(state_path, state)
        logging.info("Saved %s profile batch %s (%s profiles)", label, batch_number, len(batch))


def collect_friend_lists(client, state, user_ids, list_name, state_path):
    """Сохраняет наблюдаемые списки друзей и штатно отмечает закрытые профили"""
    lists = state[list_name]
    statuses = state["list_statuses"]
    requests_since_save = 0
    state_changed = False
    for user_id in user_ids:
        key = str(user_id)
        if key in statuses:
            continue
        profile = state["profiles"].get(key, {})
        if profile.get("is_closed"):
            lists[key] = []
            statuses[key] = "profile_closed"
            state_changed = True
            continue
        response = client.call("friends.get", {"user_id": user_id, "count": 10000})
        if response.get("__profile_closed__"):
            lists[key] = []
            statuses[key] = "profile_closed"
        elif "__error__" in response:
            code = response["__error__"].get("error_code", -1)
            lists[key] = []
            statuses[key] = f"error_{code}"
            logging.error("friends.get failed for %s: %s", user_id, response["__error__"].get("error_msg", "unknown error"))
        else:
            lists[key] = [friend_id for friend_id in response.get("items", []) if isinstance(friend_id, int)]
            statuses[key] = "loaded"
        requests_since_save += 1
        state_changed = True
        if requests_since_save == 25:
            save_state(state_path, state)
            logging.info("Saved checkpoint after 25 friends.get requests")
            requests_since_save = 0
            state_changed = False
    if state_changed:
        save_state(state_path, state)


def ordered_member_ids(ids):
    """Упорядочивает родителей согласно порядку участников группы"""
    return [member_id for member_id in GROUP_MEMBERS if member_id in ids]


def build_csv(state, output_path):
    """Собирает длинную таблицу наблюдаемых дружб со сведениями о целевой вершине"""
    member_lists = {int(user_id): friends for user_id, friends in state["member_friends"].items()}
    direct_lists = {int(user_id): friends for user_id, friends in state["direct_friends"].items()}
    profiles = {int(user_id): profile for user_id, profile in state["profiles"].items()}
    statuses = {int(user_id): status for user_id, status in state["list_statuses"].items()}
    direct_parents = {}
    for member_id, friends in member_lists.items():
        for friend_id in friends:
            direct_parents.setdefault(friend_id, set()).add(member_id)
    direct_ids = set(direct_parents)
    second_parents, second_vias = {}, {}
    rows = []
    for member_id, friends in member_lists.items():
        rows.extend((member_id, friend_id, 1) for friend_id in friends)
    for direct_id, friends in direct_lists.items():
        for friend_id in friends:
            rows.append((direct_id, friend_id, 2))
            if friend_id not in GROUP_MEMBERS and friend_id not in direct_ids:
                second_parents.setdefault(friend_id, set()).update(direct_parents.get(direct_id, set()))
                second_vias.setdefault(friend_id, set()).add(direct_id)

    def friend_row(user_id, friend_id, level):
        """Преобразует одно наблюдаемое ребро в строку согласованной схемы CSV"""
        profile = profiles.get(friend_id, {})
        parents = ordered_member_ids(direct_parents.get(friend_id, second_parents.get(friend_id, set())))
        discovery_level = 0 if friend_id in GROUP_MEMBERS else 1 if friend_id in direct_ids else 2
        return {
            "user_id": user_id,
            "friend_id": friend_id,
            "observation_level": level,
            "user_screen_name": GROUP_MEMBERS.get(user_id, profiles.get(user_id, {}).get("screen_name", "")),
            "user_is_group_member": user_id in GROUP_MEMBERS,
            "friend_first_name": profile.get("first_name", ""),
            "friend_last_name": profile.get("last_name", ""),
            "friend_screen_name": profile.get("screen_name", ""),
            "friend_is_closed": profile.get("is_closed", ""),
            "friend_deactivated": profile.get("deactivated", ""),
            "friend_profile_status": profile_status(profile),
            "friend_discovery_level": discovery_level,
            "friend_parent_ids": "|".join(map(str, parents)),
            "friend_parent_screen_name": GROUP_MEMBERS.get(parents[0], "") if parents else "",
            "friend_via_friend_ids": "|".join(map(str, sorted(second_vias.get(friend_id, set())))),
            "friend_friends_status": statuses.get(friend_id, "not_requested"),
            "friend_friends_count": len(direct_lists.get(friend_id, [])) if friend_id in direct_lists else "",
        }

    frame = pd.DataFrame([friend_row(*row) for row in rows], columns=CSV_COLUMNS).drop_duplicates(
        ["user_id", "friend_id", "observation_level"]
    )
    frame.to_csv(output_path, index=False)
    return frame, direct_ids, set(second_parents)


def run_collection(data_dir):
    """Выполняет возобновляемый конвейер и записывает итоговый CSV"""
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = data_dir / "collect_state.json"
    state = load_state(state_path)
    client = VkClient(load_tokens())
    member_ids = list(GROUP_MEMBERS)
    collect_profiles(client, state, member_ids, state_path, "member")
    collect_friend_lists(client, state, member_ids, "member_friends", state_path)
    direct_ids = sorted({friend_id for friends in state["member_friends"].values() for friend_id in friends})
    collect_profiles(client, state, direct_ids, state_path, "direct friend")
    collect_friend_lists(client, state, direct_ids, "direct_friends", state_path)
    second_ids = sorted({friend_id for friends in state["direct_friends"].values() for friend_id in friends})
    collect_profiles(client, state, second_ids, state_path, "second circle")
    frame, direct_vertices, second_vertices = build_csv(state, data_dir / "vk_friends_full.csv")
    logging.info(
        "Collection complete: level 0=%s, level 1=%s, level 2=%s, edges=%s, CSV=%s",
        len(GROUP_MEMBERS), len(direct_vertices), len(second_vertices), len(frame), data_dir / "vk_friends_full.csv",
    )


def self_check():
    """Проверяет локально схему итоговой таблицы без запроса к VK API"""
    state = empty_state()
    state["member_friends"] = {"198182178": [10]}
    state["direct_friends"] = {"10": [20]}
    state["profiles"] = {"10": {"id": 10}, "20": {"id": 20}}
    state["list_statuses"] = {"198182178": "loaded", "10": "loaded"}
    frame, direct_ids, second_ids = build_csv(state, Path("/tmp/vk_collect_self_check.csv"))
    assert list(frame.columns) == CSV_COLUMNS and direct_ids == {10} and second_ids == {20}
    Path("/tmp/vk_collect_self_check.csv").unlink()
    print("Self-check passed")


def main():
    """Разбирает параметры запуска и запускает сбор либо автономную проверку"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    arguments = parser.parse_args()
    configure_logging()
    if arguments.self_check:
        self_check()
        return
    try:
        run_collection(Path(__file__).resolve().parent / "data")
    except RuntimeError as error:
        logging.error("Collection cannot start: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
