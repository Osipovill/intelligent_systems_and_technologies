"""Общая логика проверки, построения и анализа графа VK."""

import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from config import CENTRALITY_K, CENTRALITY_BACKEND


EDGE_COLUMNS = {
    "user_id", "friend_id", "observation_level", "friend_discovery_level",
}


def prepare_edges(edges):
    """Проверяет типы рёбер и удаляет только полные дубли."""

    missing = EDGE_COLUMNS.difference(edges.columns)
    if missing:
        raise ValueError(f"Missing edge columns: {sorted(missing)}")
    prepared = edges.copy()
    integer_columns = ("user_id", "friend_id", "observation_level", "friend_discovery_level")
    for column in integer_columns:
        values = pd.to_numeric(prepared[column], errors="coerce")
        if values.isna().any() or not (values % 1 == 0).all():
            raise ValueError(f"CSV column {column} must contain integers")
        prepared[column] = values.astype("int64")
    if (prepared[["user_id", "friend_id"]] <= 0).any().any():
        raise ValueError("CSV identifiers must be positive integers")
    if not prepared["observation_level"].isin([1, 2]).all():
        raise ValueError("Observation level must be 1 or 2")
    if not prepared["friend_discovery_level"].isin([0, 1, 2]).all():
        raise ValueError("Discovery level must be 0, 1, or 2")
    return prepared.drop_duplicates().reset_index(drop=True)


def build_graph(edges, member_ids):
    """Строит неориентированный граф наблюдений и добавляет всех участников.

    Пары (a, b) и (b, a) становятся одним ребром; петли удаляются.
    Участники без наблюдаемых связей остаются изолированными вершинами.
    """
    required = {"user_id", "friend_id", "observation_level"}
    if not required.issubset(edges.columns):
        raise ValueError(f"Missing edge columns: {sorted(required - set(edges.columns))}")
    endpoints = edges[["user_id", "friend_id"]].apply(pd.to_numeric, errors="raise")
    if endpoints.isna().any().any() or (endpoints <= 0).any().any() or (endpoints % 1 != 0).any().any():
        raise ValueError("Edge endpoints must be positive integer IDs")
    if not edges["observation_level"].isin([1, 2]).all():
        raise ValueError("Observation level must be 1 or 2")
    canonical = pd.DataFrame({
        "source": endpoints.min(axis=1).astype("int64"),
        "target": endpoints.max(axis=1).astype("int64"),
    }).drop_duplicates()
    canonical = canonical.loc[canonical["source"].ne(canonical["target"])]
    graph = nx.Graph()
    graph.add_edges_from(canonical.itertuples(index=False, name=None))
    graph.add_nodes_from(member_ids)
    return graph


def graph_summary(graph, members):
    """Возвращает размеры графа и покрытие исходных участников."""

    components = list(nx.connected_components(graph))
    largest = max(components, key=len, default=set())
    member_ids = set(members)
    members_in_largest = len(member_ids & largest)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "largest_component_nodes": len(largest),
        "members": len(member_ids),
        "members_in_largest_component": members_in_largest,
        "largest_component_member_share": (
            members_in_largest / len(member_ids) if member_ids else 0.0
        ),
        "isolated_members": sum(graph.degree(uid) == 0 for uid in member_ids),
    }


def _available_cugraph():
    return "cugraph" in getattr(nx.config, "backends", {})


def calculate_betweenness(graph, *, k=CENTRALITY_K, seed=42, backend=CENTRALITY_BACKEND):
    """Считает приближённое посредничество и возвращает выбранный backend."""

    if k < 1 or backend not in {"auto", "networkx", "cugraph"}:
        raise ValueError("Positive k and backend auto/networkx/cugraph are required")
    sample_size = min(k, len(graph))
    if len(graph) < 3 or graph.number_of_edges() == 0:
        return dict.fromkeys(graph, 0.0), "networkx"
    if backend == "cugraph" and not _available_cugraph():
        raise RuntimeError("cuGraph backend is not available")
    if backend in {"auto", "cugraph"} and _available_cugraph():
        try:
            return (
                nx.betweenness_centrality(
                    graph, k=sample_size, seed=seed, normalized=True,
                    endpoints=False, backend="cugraph",
                ),
                "cugraph",
            )
        except Exception:
            if backend == "cugraph":
                raise
            logging.warning("cuGraph unavailable; calculating betweenness on CPU")
    return (
        nx.betweenness_centrality(
            graph, k=sample_size, seed=seed, normalized=True,
            endpoints=False, backend="networkx",
        ),
        "networkx",
    )


def sparse_eigenvector(graph, max_iter=3000, tol=1e-9):
    """Общий L2-нормированный вектор полного графа, включая несвязный.

    Степенной метод для A+I с одинаковым положительным начальным значением:
    при равных ведущих собственных значениях выбор воспроизводим. Не считаем
    отдельный нормированный вектор для каждой компоненты, так как тогда
    значения одногруппников из разных компонент несопоставимы.
    """
    nodes = list(graph)
    if not nodes:
        return {}
    if graph.number_of_edges() == 0:
        return dict.fromkeys(nodes, 0.0)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=float, format="csr")
    isolated = np.asarray(adjacency.sum(axis=1)).ravel() == 0
    vector = np.ones(len(nodes), dtype=float)
    vector[isolated] = 0
    vector /= np.linalg.norm(vector)
    for _ in range(max_iter):
        previous = vector
        # Умножение на A+I: сдвиг сохраняет собственные векторы A и устраняет
        # чередование знака/осцилляции на двудольных компонентах.
        vector = adjacency @ previous + previous
        # Сумма квадратов значений равна 1; сравниваем соседние итерации.
        vector /= np.linalg.norm(vector)
        if np.linalg.norm(vector - previous) < tol:
            return dict(zip(nodes, vector.tolist()))
    raise nx.PowerIterationFailedConvergence(max_iter)


def member_centralities(graph, members, *, k=CENTRALITY_K, seed=42, backend=CENTRALITY_BACKEND):
    """Возвращает DataFrame с тремя центральностями исходных участников.

    Посредничество оценивается по k случайным источникам на всём графе;
    seed делает выборку воспроизводимой для фиксированного порядка вершин.
    Близость считается точно только для участников, с поправкой на несвязность.
    Собственный вектор считается на всём графе разреженным методом выше.
    backend='auto' пробует доступный cuGraph и при ошибке переходит на CPU.
    Параметры расчёта сохраняются в frame.attrs['calculation']; запись CSV
    и сводки выполняют вызывающие ячейки блокнота.
    """
    if k < 1 or backend not in {"auto", "networkx", "cugraph"}:
        raise ValueError("Positive k and backend auto/networkx/cugraph are required")
    if not set(members).issubset(graph):
        raise ValueError("All original members must be added to the graph")
    betweenness, selected_backend = calculate_betweenness(
        graph, k=k, seed=seed, backend=backend
    )
    closeness = {uid: nx.closeness_centrality(graph, u=uid, wf_improved=True) for uid in members}
    eigenvector = sparse_eigenvector(graph)
    frame = pd.DataFrame([
        {
            "user_id": uid, "participant": name, "degree": graph.degree(uid),
            "betweenness_approximate": betweenness[uid],
            "closeness_exact_cpu": closeness[uid],
            "eigenvector_sparse_scipy": eigenvector[uid],
        }
        for uid, name in members.items()
    ])
    frame.attrs["calculation"] = {
        "nodes": len(graph), "edges": graph.number_of_edges(),
        "betweenness_k": min(k, len(graph)), "seed": seed, "backend": selected_backend,
        "networkx_version": nx.__version__,
        "closeness": "exact, Wasserman-Faust correction",
        "eigenvector": "global sparse power iteration (A+I), L2, tol=1e-9",
    }
    return frame


def betweenness_sensitivity(graph, members, *, k_values, seeds, backend="networkx"):
    """Сравнивает посредничество при разных k и случайных seeds."""

    rows = []
    for k in k_values:
        for seed in seeds:
            scores, selected_backend = calculate_betweenness(
                graph, k=k, seed=seed, backend=backend
            )
            for user_id, participant in members.items():
                rows.append({
                    "user_id": user_id,
                    "participant": participant,
                    "k": min(k, len(graph)),
                    "seed": seed,
                    "backend": selected_backend,
                    "betweenness": scores[user_id],
                })
    return pd.DataFrame(rows)


def compare_betweenness_backends(graph, members, *, k=CENTRALITY_K, seed=42, cpu_scores=None):
    """Сравнивает CPU и GPU для посредничества, если GPU доступен."""

    if cpu_scores is None:
        cpu_scores, _ = calculate_betweenness(graph, k=k, seed=seed, backend="networkx")
    rows = [
        {
            "backend": "networkx",
            "status": "available",
            "participant": participant,
            "betweenness": cpu_scores[user_id],
            "absolute_difference_from_cpu": 0.0,
        }
        for user_id, participant in members.items()
    ]
    try:
        gpu_scores, selected_backend = calculate_betweenness(
            graph, k=k, seed=seed, backend="cugraph"
        )
    except Exception as error:
        rows.append({
            "backend": "cugraph",
            "status": f"unavailable: {error}",
            "participant": "",
            "betweenness": np.nan,
            "absolute_difference_from_cpu": np.nan,
        })
    else:
        rows.extend({
            "backend": selected_backend,
            "status": "available",
            "participant": participant,
            "betweenness": gpu_scores[user_id],
            "absolute_difference_from_cpu": abs(gpu_scores[user_id] - cpu_scores[user_id]),
        } for user_id, participant in members.items())
    return pd.DataFrame(rows)


def leader_gaps(frame, metrics, *, participant_column="participant"):
    """Возвращает лидера, второе место и разрыв по каждой метрике."""

    rows = []
    for metric in metrics:
        ordered = frame.sort_values(
            [metric, participant_column], ascending=[False, True]
        ).reset_index(drop=True)
        rows.append({
            "metric": metric,
            "leader": ordered.loc[0, participant_column],
            "leader_value": ordered.loc[0, metric],
            "second": ordered.loc[1, participant_column] if len(ordered) > 1 else "",
            "second_value": ordered.loc[1, metric] if len(ordered) > 1 else np.nan,
            "gap": ordered.loc[0, metric] - ordered.loc[1, metric] if len(ordered) > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def mark_analysis_complete(summary_path: Path, calculation: dict, graph_stats: dict):
    """Сохраняет факт завершения анализа и его параметры в сводке снимка."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["analysis_completed"] = True
    summary["analysis"] = {"calculation": calculation, "graph": graph_stats}
    temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
