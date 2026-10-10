"""Анализ сохранённого графа VK без токенов и сетевых запросов."""

import logging

import networkx as nx
import numpy as np
import pandas as pd


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


def member_centralities(graph, members, *, k=128, seed=42, backend="auto"):
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
    sample_size = min(k, len(graph))
    selected_backend = "networkx"
    if len(graph) < 3 or graph.number_of_edges() == 0:
        betweenness = dict.fromkeys(graph, 0.0)
    else:
        betweenness = None
        if backend == "cugraph" or (backend == "auto" and "cugraph" in nx.config.backends):
            try:
                betweenness = nx.betweenness_centrality(
                    graph, k=sample_size, seed=seed, normalized=True,
                    endpoints=False, backend="cugraph",
                )
                selected_backend = "cugraph"
            except Exception:
                if backend == "cugraph":
                    raise
                logging.warning("cuGraph unavailable; calculating betweenness on CPU")
        if betweenness is None:
            betweenness = nx.betweenness_centrality(
                graph, k=sample_size, seed=seed, normalized=True,
                endpoints=False, backend="networkx",
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
        "betweenness_k": sample_size, "seed": seed, "backend": selected_backend,
        "networkx_version": nx.__version__,
        "closeness": "exact, Wasserman-Faust correction",
        "eigenvector": "global sparse power iteration (A+I), L2, tol=1e-9",
    }
    return frame
