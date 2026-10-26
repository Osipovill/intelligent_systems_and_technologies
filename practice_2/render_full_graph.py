"""Сохраняет полный граф VK со схематической радиальной GPU-раскладкой"""

import argparse
from collections import deque
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

from vk_analysis import build_graph


CATEGORY_COLORS = {
    "members": "#1f77b4",
    "friends": "#ff7f0e",
    "friends_of_friends": "#bdbdbd",
}


def node_categories(graph, members):
    """Делит весь граф на исходников, прямых друзей и оставшийся второй круг"""
    originals = set(members)
    if not originals.issubset(graph):
        raise ValueError("Members must belong to graph")
    friends = {
        neighbor for member in originals for neighbor in graph[member]
    } - originals
    return {
        "members": originals,
        "friends": friends,
        "friends_of_friends": set(graph) - originals - friends,
    }


def require_gpu():
    """Проверяет реальное вычисление CUDA и не скрывает отказ GPU"""
    try:
        import cupy as cp

        cp.zeros(1).sum().item()
        return cp
    except Exception as error:
        raise RuntimeError(
            "GPU backend unavailable; run Docker with --gpus all"
        ) from error


def radial_layout(graph, members, *, xp):
    """Размещает весь BFS-лес по интервалам листьев за линейное время"""
    nodes = list(graph)
    if not set(members).issubset(graph):
        raise ValueError("Members must belong to graph")
    if not nodes:
        return nodes, np.empty((0, 2), dtype=np.float32)
    index = {node: i for i, node in enumerate(nodes)}
    children = [[] for _ in nodes]
    depth = np.zeros(len(nodes), dtype=np.int32)
    roots = list(dict.fromkeys(members))
    seen = set(roots)
    order = []
    queue = deque(roots)

    # Участники одновременно задают корни, оставшиеся компоненты добавляются целиком
    for candidate in nodes:
        if not queue:
            if candidate in seen:
                continue
            roots.append(candidate)
            seen.add(candidate)
            queue.append(candidate)
        while queue:
            node = queue.popleft()
            parent_index = index[node]
            order.append(parent_index)
            for neighbor in graph[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
                    child_index = index[neighbor]
                    children[parent_index].append(child_index)
                    depth[child_index] = depth[parent_index] + 1

    weights = np.ones(len(nodes), dtype=np.int64)
    for parent in reversed(order):
        if children[parent]:
            weights[parent] = sum(weights[child] for child in children[parent])
    starts = np.zeros(len(nodes), dtype=np.int64)
    total = 0
    for root in roots:
        root_index = index[root]
        starts[root_index] = total
        total += int(weights[root_index])
    for parent in order:
        cursor = starts[parent]
        for child in children[parent]:
            starts[child] = cursor
            cursor += weights[child]

    # GPU вычисляет углы и координаты, обход и последующий PNG остаются на CPU
    angle = (xp.asarray(starts, dtype=xp.float64) + xp.asarray(weights) / 2) * (
        2 * xp.pi / total
    )
    radius = xp.asarray(depth, dtype=xp.float64) + 1
    positions = xp.stack((radius * xp.cos(angle), radius * xp.sin(angle)), axis=1)
    if xp is not np:
        positions = xp.asnumpy(positions)
    return nodes, positions


def render_graph(graph, members, output, *, xp, pixels=4000):
    """Рисует все рёбра и вершины и сохраняет проверяемую JSON-сводку"""
    output = Path(output)
    if output.suffix.lower() != ".png" or not 128 <= pixels <= 8192:
        raise ValueError("Output must be PNG; pixels must be between 128 and 8192")
    started = perf_counter()
    nodes, positions = radial_layout(graph, members, xp=xp)
    layout_seconds = perf_counter() - started
    index = {node: i for i, node in enumerate(nodes)}
    endpoints = np.asarray(
        [(index[a], index[b]) for a, b in graph.edges], dtype=np.int64
    ).reshape(-1, 2)
    segments = positions[endpoints]
    categories = node_categories(graph, members)
    category_counts = {category: len(group) for category, group in categories.items()}
    member_indices = [index[node] for node in members]
    fig, ax = plt.subplots(
        figsize=(pixels / 200, pixels / 200), dpi=200, facecolor="#0b1020"
    )
    try:
        ax.set_facecolor("#0b1020")
        ax.add_collection(
            LineCollection(segments, colors="#829ac0", linewidths=0.12, alpha=0.12)
        )
        # Более редкие категории рисуются поверх второго круга без повторения узлов
        for category, label, size in (
            ("friends_of_friends", "Друзья друзей", 0.6),
            ("friends", "Прямые друзья", 5),
            ("members", "Исходники", 32),
        ):
            selected = positions[
                [i for i, node in enumerate(nodes) if node in categories[category]]
            ]
            ax.scatter(
                selected[:, 0],
                selected[:, 1],
                s=size,
                c=CATEGORY_COLORS[category],
                alpha=0.9,
                linewidths=0,
                label=f"{label}: {category_counts[category]:,}",
            )
        if member_indices:
            selected = positions[member_indices]
            for node, point in zip(members, selected):
                ax.annotate(
                    str(members[node]),
                    point,
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=6,
                    color="white",
                )
        legend = ax.legend(
            loc="upper right",
            facecolor="#0b1020",
            edgecolor="#829ac0",
            labelcolor="white",
            fontsize=8,
        )
        for handle in legend.legend_handles:
            handle.set_sizes([30])
        ax.set_title(
            f"Полный наблюдаемый граф VK: {len(nodes):,} вершин, {len(endpoints):,} рёбер",
            color="white",
            fontsize=13,
        )
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.08)
        ax.set_axis_off()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    report = {
        "nodes": len(nodes),
        "edges": graph.number_of_edges(),
        "members": len(members),
        "category_counts": category_counts,
        "category_colors": CATEGORY_COLORS,
        "rendered_nodes": len(positions),
        "rendered_edges": len(segments),
        "layout": "schematic multi-source BFS radial forest; no sampling",
        "layout_backend": "cupy-cuda" if xp is not np else "numpy-test",
        "render_backend": "matplotlib-agg-cpu",
        "pixels": pixels,
        "layout_seconds": round(layout_seconds, 3),
        "render_total_seconds": round(perf_counter() - started, 3),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    """Читает полные локальные CSV и запускает обязательный GPU backend"""
    directory = Path(__file__).resolve().parent / "data" / "group"
    parser = argparse.ArgumentParser(
        description="Render every node and edge of the saved VK graph using a schematic GPU layout"
    )
    parser.add_argument("--data-dir", type=Path, default=directory)
    parser.add_argument("--output", type=Path, default=directory / "full_graph.png")
    parser.add_argument("--pixels", type=int, default=4000)
    args = parser.parse_args()
    started = perf_counter()
    xp = require_gpu()
    edges = pd.read_csv(
        args.data_dir / "vk_friends_full.csv",
        usecols=["user_id", "friend_id", "observation_level"],
    )
    member_frame = pd.read_csv(
        args.data_dir / "members.csv", usecols=["user_id", "screen_name"]
    )
    members = dict(zip(member_frame["user_id"], member_frame["screen_name"]))
    graph = build_graph(edges, members)
    report = render_graph(graph, members, args.output, xp=xp, pixels=args.pixels)
    report["total_seconds"] = round(perf_counter() - started, 3)
    report["gpu"] = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"Saved image: {args.output}")


if __name__ == "__main__":
    main()
