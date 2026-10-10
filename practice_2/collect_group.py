"""Сбор двух кругов друзей всех одногруппников из data/group_ids.csv."""

import argparse
import csv
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# Известные ID команды помогают переиспользовать её кеш при разрешении ссылок.
TEAM_MEMBERS = {
    198182178: "flashik12", 473310225: "0pupok00",
    424935969: "ryosuk3", 465105120: "medvediano",
}
API_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"
PROFILE_FIELDS = "screen_name"
PROFILE_BATCH_SIZE = 500
FRIENDS_PAGE_SIZE = 5000
REQUEST_INTERVAL = 0.5
TERMINAL_STATUSES = {"loaded", "profile_closed", "access_denied", "deactivated"}
CSV_COLUMNS = [
    "user_id", "friend_id", "observation_level", "user_screen_name",
    "user_is_group_member", "friend_first_name", "friend_last_name",
    "friend_screen_name", "friend_is_closed", "friend_deactivated",
    "friend_profile_status", "friend_discovery_level", "friend_parent_ids",
    "friend_parent_screen_name", "friend_via_friend_ids",
    "friend_friends_status", "friend_friends_count",
]
MEMBER_COLUMNS = [
    "user_id", "input_ref", "screen_name", "first_name", "last_name",
    "is_closed", "deactivated", "friends_status", "friends_count", "list_source",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class CollectionPaused(RuntimeError):
    """Прогресс сохранён; повторный запуск продолжит сбор."""


class VkApiError(RuntimeError):
    def __init__(self, code):
        self.code = code
        # Не сохраняем request_params и сообщения исключений HTTP с токеном.
        super().__init__(f"VK API error {code}")


def load_tokens(env_path=None):
    load_dotenv(env_path or BASE_DIR / ".env", override=False)
    raw = os.environ.get("VK_SERVICE_TOKENS") or os.environ.get("VK_SERVICE_TOKEN", "")
    tokens = list(dict.fromkeys(token.strip() for token in raw.split(",") if token.strip()))
    if not tokens:
        raise CollectionPaused("Set VK_SERVICE_TOKEN or VK_SERVICE_TOKENS in .env/environment")
    return tokens


class VkClient:
    """Последовательные POST-запросы, общий темп, ограниченные повторы."""

    def __init__(self, tokens, interval=REQUEST_INTERVAL, max_attempts=5, max_requests=None):
        if not tokens or interval <= 0 or max_attempts < 1:
            raise ValueError("Tokens, positive interval and max_attempts are required")
        self.tokens = list(tokens)
        self.interval = interval
        self.max_attempts = max_attempts
        self.max_requests = max_requests
        self.request_count = 0
        self.last_request_at = 0.0
        self.session = requests.Session()

    def call(self, method, params):
        for attempt in range(self.max_attempts):
            if self.max_requests is not None and self.request_count >= self.max_requests:
                raise CollectionPaused("Request budget exhausted")
            wait = self.interval - (time.monotonic() - self.last_request_at)
            if wait > 0:
                time.sleep(wait)
            self.last_request_at = time.monotonic()
            self.request_count += 1
            try:
                response = self.session.post(
                    API_URL + method,
                    data={**params, "access_token": self.tokens[0], "v": API_VERSION},
                    timeout=(10, 30),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Unexpected response type")
            except (requests.RequestException, ValueError):
                logging.warning("%s: HTTP/network/JSON failure, attempt %d", method, attempt + 1)
            else:
                if "response" in payload:
                    return payload["response"]
                code = payload.get("error", {}).get("error_code", -1)
                if code == 5:
                    if len(self.tokens) == 1:
                        raise CollectionPaused("VK token is invalid (error 5)")
                    self.tokens.pop(0)
                    continue
                if code in (9, 29):
                    # Flood control / quota: прекращаем запуск, не обходим ротацией.
                    raise CollectionPaused(f"VK flood control or quota exhausted (error {code})")
                if code in (14, 17, 27, 28):
                    raise CollectionPaused(f"VK requires authentication/validation (error {code})")
                if code not in (1, 6, 10):
                    raise VkApiError(code)
                logging.warning("%s: VK error %s, attempt %d", method, code, attempt + 1)
            if attempt + 1 < self.max_attempts:
                time.sleep(min(60, 2 ** (attempt + 1)))
        raise CollectionPaused(f"{method}: retry budget exhausted; restart later")


def empty_state():
    return {
        "member_friends": {}, "direct_friends": {}, "profiles": {}, "list_statuses": {},
        "list_sources": {}, "friend_pages": {}, "resolved_refs": {},
    }


def load_state(path):
    if not path.exists():
        return empty_state()
    state = json.loads(path.read_text(encoding="utf-8"))
    for key, default in empty_state().items():
        state.setdefault(key, default)
    logging.info("Loaded checkpoint %s", path)
    return state


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    replace_file(temporary, path)


def replace_file(temporary, destination):
    """Windows: читатель/антивирус может ненадолго удерживать файл открытым."""
    for attempt in range(10):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def seed_cache(state, path):
    """Копирует кеш; исходная выгрузка команды остаётся отдельным снимком."""
    cache = load_state(path)
    for key, profile in cache["profiles"].items():
        if not profile.get("_profile_status", "loaded").startswith("error_"):
            state["profiles"].setdefault(key, profile)
    for name in ("member_friends", "direct_friends"):
        for key, friends in cache[name].items():
            status = cache["list_statuses"].get(key)
            if status in TERMINAL_STATUSES and key not in state["list_statuses"]:
                state["direct_friends"][key] = friends
                state["list_statuses"][key] = status
                state["list_sources"][key] = f"cache:{path.name}"
    record_cache_source(state, path)


def record_cache_source(state, path):
    # Переносимый путь и дата исходного снимка вместо локального абсолютного пути.
    state["cache_source"] = os.path.relpath(path, BASE_DIR).replace(os.sep, "/")
    log_path = path.parent.parent / "collect.log"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        dates = [line[:19] for line in lines if "Collection complete:" in line]
        if dates:
            state["cache_collected_at"] = dates[-1]


def normalize_ref(value):
    value = value.strip()
    if "://" in value:
        parsed = urlparse(value)
        if parsed.hostname not in {"vk.com", "vk.ru", "www.vk.com", "www.vk.ru", "m.vk.com", "m.vk.ru"}:
            raise ValueError(f"Not a VK profile URL: {value}")
        value = parsed.path.strip("/")
    if not re.fullmatch(r"[A-Za-z0-9_.]+", value):
        raise ValueError(f"Invalid VK reference: {value}")
    if value.isdigit() and int(value) <= 0:
        raise ValueError("User IDs must be positive")
    return value.lower()


def read_member_refs(path):
    """Читает один столбец URL/ID/никнеймов с необязательным заголовком."""
    refs = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.reader(file):
            if not row or not any(value.strip() for value in row):
                continue
            if len(row) != 1:
                raise ValueError("Member CSV must have exactly one column")
            if not refs and row[0].strip().lower() in {"user_id", "url", "vk_url", "screen_name"}:
                continue
            ref = normalize_ref(row[0])
            if ref not in refs:
                refs.append(ref)
    if not refs:
        raise ValueError("Member CSV is empty")
    return refs


def resolve_members(client, state, refs, state_path):
    known = {
        str(profile.get("screen_name") or "").lower(): int(key)
        for key, profile in state["profiles"].items()
    }
    known.update({name: user_id for user_id, name in TEAM_MEMBERS.items()})
    members = {}
    for ref in refs:
        numeric = re.fullmatch(r"(?:id)?([0-9]+)", ref)
        user_id = state["resolved_refs"].get(ref) or known.get(ref)
        if numeric:
            user_id = int(numeric[1])
        if not user_id:
            result = client.call("utils.resolveScreenName", {"screen_name": ref})
            if not isinstance(result, dict) or result.get("type") != "user":
                raise ValueError(f"Reference does not resolve to a user: {ref}")
            user_id = result["object_id"]
        if user_id <= 0:
            raise ValueError(f"Invalid user ID for {ref}")
        state["resolved_refs"][ref] = user_id
        members.setdefault(user_id, ref)
        save_state(state_path, state)
    return members


def collect_profiles(client, state, user_ids, state_path, label):
    missing = [uid for uid in sorted(set(user_ids))
               if str(uid) not in state["profiles"]
               or state["profiles"][str(uid)].get("_profile_status", "").startswith("error_")]
    for start in range(0, len(missing), PROFILE_BATCH_SIZE):
        batch = missing[start:start + PROFILE_BATCH_SIZE]
        response = client.call("users.get", {"user_ids": ",".join(map(str, batch)), "fields": PROFILE_FIELDS})
        if not isinstance(response, list):
            raise CollectionPaused("Unexpected users.get response")
        received = {profile["id"]: profile for profile in response}
        for uid in batch:
            state["profiles"][str(uid)] = received.get(uid, {"id": uid, "_profile_status": "not_returned"})
        save_state(state_path, state)
        logging.info("%s: profiles %d/%d", label, min(start + len(batch), len(missing)), len(missing))


def collect_friend_lists(client, state, user_ids, list_name, state_path):
    for index, uid in enumerate(user_ids, 1):
        key = str(uid)
        status = state["list_statuses"].get(key)
        if status in TERMINAL_STATUSES:
            other = "direct_friends" if list_name == "member_friends" else "member_friends"
            if key not in state[list_name] and key in state[other]:
                state[list_name][key] = state[other][key]
            if key in state[list_name]:
                continue
        profile = state["profiles"].get(key, {})
        if profile.get("deactivated"):
            status, friends = "deactivated", []
        elif profile.get("is_closed") and not profile.get("can_access_closed"):
            status, friends = "profile_closed", []
        else:
            page = state["friend_pages"].setdefault(key, {"items": [], "offset": 0})
            status = "loaded"
            while True:
                try:
                    response = client.call("friends.get", {
                        "user_id": uid, "count": FRIENDS_PAGE_SIZE, "offset": page["offset"],
                    })
                except VkApiError as error:
                    statuses = {15: "access_denied", 18: "deactivated", 30: "profile_closed"}
                    if error.code not in statuses:
                        raise
                    status = statuses[error.code]
                    page["items"] = []
                    break
                if not isinstance(response, dict) or "items" not in response or "count" not in response:
                    raise CollectionPaused("Unexpected friends.get response")
                items = response["items"]
                if any(type(item) is not int or item <= 0 for item in items):
                    raise CollectionPaused("friends.get returned invalid IDs")
                page["items"].extend(items)
                page["offset"] += len(items)
                if page["offset"] >= response["count"]:
                    if len(set(page["items"])) != response["count"]:
                        state["friend_pages"].pop(key)
                        raise CollectionPaused(f"Friend list changed during pagination for {uid}; restart")
                    break
                if not items:
                    raise CollectionPaused(f"Incomplete pagination for {uid}")
                save_state(state_path, state)
            friends = sorted(set(page["items"]))
        state[list_name][key] = friends
        state["list_statuses"][key] = status
        state["list_sources"][key] = f"api:{utc_now()}"
        state["friend_pages"].pop(key, None)
        save_state(state_path, state)
        logging.info("%s %d/%d: %s, %d friends (%s)", list_name, index, len(user_ids), uid, len(friends), status)


def scoped_lists(state, members):
    member_lists = {uid: state["member_friends"].get(str(uid), []) for uid in members}
    direct_ids = {fid for friends in member_lists.values() for fid in friends}
    direct_lists = {
        uid: state["direct_friends"].get(str(uid), state["member_friends"].get(str(uid), []))
        for uid in sorted(direct_ids - set(members))
    }
    return member_lists, direct_lists, direct_ids


def atomic_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    replace_file(temporary, path)


def build_csv(state, output_path, members):
    """Пишет прежнюю схему CSV потоком; кеш чужих списков не попадает в граф."""
    member_lists, direct_lists, direct_ids = scoped_lists(state, members)
    names = {uid: state["profiles"].get(str(uid), {}).get("screen_name", ref) for uid, ref in members.items()}
    parents, second_parents, second_vias = {}, {}, {}
    for uid, friends in member_lists.items():
        for fid in friends:
            parents.setdefault(fid, set()).add(uid)
    for uid, friends in direct_lists.items():
        for fid in friends:
            if fid not in members and fid not in direct_ids:
                second_parents.setdefault(fid, set()).update(parents.get(uid, set()))
                second_vias.setdefault(fid, set()).add(uid)
    count = 0

    def rows():
        nonlocal count
        for level, lists in ((1, member_lists), (2, direct_lists)):
            for uid, friends in lists.items():
                for fid in sorted(set(friends)):
                    if uid == fid:
                        continue
                    profile = state["profiles"].get(str(fid), {})
                    parent_ids = [mid for mid in members if mid in parents.get(fid, second_parents.get(fid, set()))]
                    requested = fid in members or fid in direct_ids
                    status = state["list_statuses"].get(str(fid), "not_requested") if requested else "not_requested"
                    friend_list = state["member_friends"].get(str(fid), state["direct_friends"].get(str(fid), []))
                    count += 1
                    yield {
                        "user_id": uid, "friend_id": fid, "observation_level": level,
                        "user_screen_name": names.get(uid, state["profiles"].get(str(uid), {}).get("screen_name", "")),
                        "user_is_group_member": uid in members,
                        "friend_first_name": profile.get("first_name", ""),
                        "friend_last_name": profile.get("last_name", ""),
                        "friend_screen_name": profile.get("screen_name", ""),
                        "friend_is_closed": profile.get("is_closed", ""),
                        "friend_deactivated": profile.get("deactivated", ""),
                        "friend_profile_status": profile.get("_profile_status", "loaded" if profile else "not_loaded"),
                        "friend_discovery_level": 0 if fid in members else 1 if fid in direct_ids else 2,
                        "friend_parent_ids": "|".join(map(str, parent_ids)),
                        "friend_parent_screen_name": names[parent_ids[0]] if parent_ids else "",
                        "friend_via_friend_ids": "|".join(map(str, sorted(second_vias.get(fid, set())))),
                        "friend_friends_status": status,
                        "friend_friends_count": len(friend_list) if status == "loaded" else "",
                    }

    atomic_csv(output_path, CSV_COLUMNS, rows())
    return count, direct_ids, set(second_parents)


def export_collection(state, data_dir, members, completed, client):
    rows, direct_ids, second_ids = build_csv(state, data_dir / "vk_friends_full.csv", members)
    requested_ids = set(members) | direct_ids
    graph_completed = all(
        state["list_statuses"].get(str(uid)) in TERMINAL_STATUSES
        for uid in requested_ids
    )
    pending_profiles = sum(
        str(uid) not in state["profiles"]
        or state["profiles"][str(uid)].get("_profile_status", "").startswith("error_")
        for uid in requested_ids | second_ids
    )
    profiles_completed = pending_profiles == 0
    member_rows = []
    for uid, ref in members.items():
        profile = state["profiles"].get(str(uid), {})
        status = state["list_statuses"].get(str(uid), "not_requested")
        member_rows.append({
            "user_id": uid, "input_ref": ref, "screen_name": profile.get("screen_name", ref),
            "first_name": profile.get("first_name", ""), "last_name": profile.get("last_name", ""),
            "is_closed": profile.get("is_closed", ""), "deactivated": profile.get("deactivated", ""),
            "friends_status": status,
            "friends_count": len(state["member_friends"].get(str(uid), [])) if status == "loaded" else "",
            "list_source": state["list_sources"].get(str(uid), "legacy_checkpoint"),
        })
    atomic_csv(data_dir / "members.csv", MEMBER_COLUMNS, member_rows)
    save_state(data_dir / "collection_summary.json", {
        "completed": completed, "graph_completed": graph_completed,
        "profiles_completed": profiles_completed,
        "pending_profiles": pending_profiles,
        "exported_at": utc_now(), "api_version": API_VERSION,
        "members": len(members), "direct_ids_including_members": len(direct_ids),
        "second_circle": len(second_ids), "observations": rows,
        "requests_this_run": client.request_count, "request_interval": client.interval,
        "cache_source": state.get("cache_source"),
        "cache_collected_at": state.get("cache_collected_at"),
        "note": "Cached lists retain their original snapshot; unavailable lists are not empty friend lists.",
    })
    logging.info("Exported snapshot: members=%d, direct=%d, second=%d, observations=%d, completed=%s",
                 len(members), len(direct_ids), len(second_ids), rows, completed)


def run_collection(data_dir, members_csv, cache_from=None, interval=REQUEST_INTERVAL,
                   max_attempts=5, max_requests=None, env_path=None, retry_unavailable=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = data_dir / "collect_state.json"
    state = load_state(state_path)
    refs = read_member_refs(members_csv)
    if state.get("member_refs", refs) != refs:
        raise ValueError("Checkpoint belongs to a different member list; choose a new --data-dir")
    state["member_refs"] = refs
    if cache_from and not state.get("cache_source"):
        seed_cache(state, cache_from)
    elif cache_from:
        record_cache_source(state, cache_from)
    if retry_unavailable:
        for key in list(state["list_statuses"]):
            if state["list_statuses"][key] != "loaded":
                state["list_statuses"].pop(key)
                state["profiles"].pop(key, None)
    save_state(state_path, state)
    client = VkClient(load_tokens(env_path), interval, max_attempts, max_requests)
    members = {}
    completed = False
    try:
        members = resolve_members(client, state, refs, state_path)
        collect_profiles(client, state, members, state_path, "members")
        collect_friend_lists(client, state, list(members), "member_friends", state_path)
        _, _, direct_ids = scoped_lists(state, members)
        collect_profiles(client, state, direct_ids, state_path, "direct circle")
        collect_friend_lists(client, state, sorted(direct_ids - set(members)), "direct_friends", state_path)
        _, direct_lists, _ = scoped_lists(state, members)
        second_ids = {fid for friends in direct_lists.values() for fid in friends}
        collect_profiles(client, state, second_ids, state_path, "second circle")
        completed = True
    finally:
        save_state(state_path, state)
        if members:
            export_collection(state, data_dir, members, completed, client)
        client.session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members-csv", type=Path, default=BASE_DIR / "data" / "group_ids.csv")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data" / "group")
    parser.add_argument("--cache-from", type=Path, default=BASE_DIR / "data" / "collect_state.json")
    parser.add_argument("--env-file", type=Path, default=BASE_DIR / ".env")
    parser.add_argument("--interval", type=float, default=REQUEST_INTERVAL)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--retry-unavailable", action="store_true")
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.max_attempts < 1 or (args.max_requests is not None and args.max_requests < 1):
        parser.error("interval/max-attempts/max-requests must be positive")
    if args.members_csv and args.data_dir.resolve() == (BASE_DIR / "data").resolve():
        parser.error("Use a separate --data-dir for a custom member list")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_collection(args.data_dir, args.members_csv, args.cache_from, args.interval,
                       args.max_attempts, args.max_requests, args.env_file, args.retry_unavailable)
    except (CollectionPaused, VkApiError, ValueError, OSError) as error:
        logging.error("Collection stopped: %s", error)
        return 1
    except KeyboardInterrupt:
        logging.warning("Interrupted; checkpoint and partial CSV saved")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
