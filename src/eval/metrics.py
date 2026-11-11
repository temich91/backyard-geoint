"""Метрики качества визуального поиска мест (Visual Place Recognition)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Вычисление расстояния по дуге большого круга (в метрах) по формуле гаверсинусов."""
    earth_radius_m = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    sin_half_dphi = math.sin(delta_phi / 2.0)
    sin_half_dlambda = math.sin(delta_lambda / 2.0)

    a = (
        sin_half_dphi * sin_half_dphi
        + math.cos(phi1) * math.cos(phi2) * sin_half_dlambda * sin_half_dlambda
    )
    # Защита от погрешностей округления с плавающей точкой
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return earth_radius_m * c


def compute_recall_at_k(
    gt_coords: Sequence[tuple[float, float]],
    retrieved_coords: Sequence[Sequence[tuple[float, float]]],
    k_values: Sequence[int] = (1, 5, 10),
    radius_m: float = 50.0,
) -> dict[int, float]:
    """Доля запросов (0.0 .. 1.0), для которых хотя бы один из топ-k кандидатов находится в пределах radius_m.

    Параметры:
        gt_coords: список (lat, lon) истинных положений запросов длины Q.
        retrieved_coords: список списков (lat, lon) найденных кандидатов длины Q.
        k_values: значения k (например, 1, 5, 10).
        radius_m: радиус попадания в метрах (например, 25.0 или 50.0).
    """
    total_queries = len(gt_coords)
    if total_queries == 0:
        return {k: 0.0 for k in k_values}

    hits_at_k = {k: 0 for k in k_values}

    for (gt_lat, gt_lon), candidates in zip(gt_coords, retrieved_coords):
        distances = [
            haversine_distance(gt_lat, gt_lon, c_lat, c_lon)
            for c_lat, c_lon in candidates
        ]
        for k in k_values:
            top_k_distances = distances[:k]
            if top_k_distances and min(top_k_distances) <= radius_m:
                hits_at_k[k] += 1

    return {k: hits_at_k[k] / total_queries for k in k_values}


def compute_localization_errors(
    gt_coords: Sequence[tuple[float, float]],
    retrieved_coords: Sequence[Sequence[tuple[float, float]]],
) -> dict[str, float]:
    """Статистика географической ошибки (в метрах) для топ-1 кандидата."""
    total_queries = len(gt_coords)
    if total_queries == 0:
        return {
            "mean_m": 0.0,
            "median_m": 0.0,
            "min_m": 0.0,
            "max_m": 0.0,
            "p25_m": 0.0,
            "p75_m": 0.0,
            "p90_m": 0.0,
        }

    errors: list[float] = []
    for (gt_lat, gt_lon), candidates in zip(gt_coords, retrieved_coords):
        if candidates:
            c_lat, c_lon = candidates[0]
            errors.append(haversine_distance(gt_lat, gt_lon, c_lat, c_lon))
        else:
            errors.append(float("inf"))

    sorted_errors = sorted(errors)
    n = len(sorted_errors)

    def percentile(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        weight = idx - lower
        return sorted_errors[lower] * (1.0 - weight) + sorted_errors[upper] * weight

    return {
        "mean_m": sum(errors) / n,
        "median_m": percentile(50.0),
        "min_m": sorted_errors[0],
        "max_m": sorted_errors[-1],
        "p25_m": percentile(25.0),
        "p75_m": percentile(75.0),
        "p90_m": percentile(90.0),
    }


def evaluate_results(
    queries_metadata: Sequence[Mapping[str, Any]],
    search_matches: Sequence[Sequence[Any]],
    radii_m: Sequence[float] = (25.0, 50.0),
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Комплексная оценка качества по метаданным запросов и результатам поиска."""
    if len(queries_metadata) != len(search_matches):
        raise ValueError(
            f"Число запросов в метаданных ({len(queries_metadata)}) "
            f"не совпадает с числом результатов поиска ({len(search_matches)})"
        )

    gt_coords: list[tuple[float, float]] = []
    for i, q in enumerate(queries_metadata):
        try:
            gt_coords.append((float(q["lat"]), float(q["lon"])))
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Запрос {i}: отсутствуют или некорректны координаты lat/lon") from e

    retrieved_coords: list[list[tuple[float, float]]] = []
    for i, matches in enumerate(search_matches):
        query_candidates: list[tuple[float, float]] = []
        for match in matches:
            meta = match.metadata if hasattr(match, "metadata") else match.get("metadata", match)
            query_candidates.append((float(meta["lat"]), float(meta["lon"])))
        retrieved_coords.append(query_candidates)

    recall_metrics: dict[str, dict[str, float]] = {}
    for r in radii_m:
        recalls = compute_recall_at_k(gt_coords, retrieved_coords, k_values=k_values, radius_m=r)
        recall_metrics[f"within_{int(r)}m"] = {f"recall@{k}": v for k, v in recalls.items()}

    error_metrics = compute_localization_errors(gt_coords, retrieved_coords)

    return {
        "num_queries": len(gt_coords),
        "recall": recall_metrics,
        "localization_error": error_metrics,
    }
