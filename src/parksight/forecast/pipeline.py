from __future__ import annotations

import gc
import json
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch import nn

from parksight import config
from parksight.forecast import baselines, evaluate, graph, windows
from parksight.forecast.model import SpatioTemporalGCN, TemporalEncoder
from parksight.ingest import loader

MAX_NODES = 1500
LOOKBACK = 21
HORIZON = 7
NEIGHBOURS = 8
HIDDEN = 32
GRU_HIDDEN = 32
EPOCHS = 60
LEARNING_RATE = 0.01
TOP_K = 50


@dataclass
class Result:
    metrics: dict
    forecast: dict[str, float]


def run(
    source: str | None = None, seed: int = 0, with_graph: bool = True, persist: bool = True
) -> Result:
    torch.manual_seed(seed)
    np.random.seed(seed)

    frame = loader.load_violations(config.source_path(source) if source else config.RAW_DATA_PATH)
    cells = graph.select_cells(frame, MAX_NODES)
    counts, severity, dates, coords = graph.daily_tensor(frame, cells)
    del frame
    gc.collect()
    features = graph.feature_tensor(counts, severity, dates)
    edge_index_np, edge_weight_np = graph.knn_edges(coords, NEIGHBOURS)

    features_t = torch.from_numpy(features)
    counts_t = torch.from_numpy(counts)
    edge_index = torch.from_numpy(edge_index_np)
    edge_weight = torch.from_numpy(edge_weight_np)

    starts = windows.window_starts(len(dates), LOOKBACK, HORIZON)
    train_idx, val_idx, test_idx = windows.time_split(len(starts))
    train_starts = [starts[i] for i in train_idx]
    val_starts = [starts[i] for i in val_idx]
    test_starts = [starts[i] for i in test_idx]

    def sequence(start: int) -> torch.Tensor:
        return features_t[start - LOOKBACK : start]

    def future(start: int) -> torch.Tensor:
        return counts_t[start : start + HORIZON].sum(dim=0)

    def prior(start: int) -> torch.Tensor:
        return counts_t[start - HORIZON : start].sum(dim=0)

    train_residuals = np.stack(
        [counts[s : s + HORIZON].sum(0) - counts[s - HORIZON : s].sum(0) for s in train_starts]
    )
    residual_mean = float(train_residuals.mean())
    residual_std = float(train_residuals.std()) + 1e-6

    actual_val = np.stack([future(s).numpy() for s in val_starts])
    actual = np.stack([future(s).numpy() for s in test_starts])

    def fit(make_net):
        net = make_net()
        optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        objective = nn.SmoothL1Loss()

        def predict(window_starts):
            net.eval()
            with torch.no_grad():
                rows = []
                for start in window_starts:
                    standardised = net(sequence(start), edge_index, edge_weight)
                    residual = standardised * residual_std + residual_mean
                    rows.append((prior(start) + residual).clamp(min=0))
            return torch.stack(rows).numpy()

        best = float("inf")
        best_state = None
        for _ in range(EPOCHS):
            net.train()
            for index in np.random.permutation(len(train_starts)):
                start = train_starts[index]
                optimiser.zero_grad()
                prediction = net(sequence(start), edge_index, edge_weight)
                target = (future(start) - prior(start) - residual_mean) / residual_std
                loss = objective(prediction, target)
                loss.backward()
                optimiser.step()
            score = evaluate.mae(actual_val, predict(val_starts))
            if score < best:
                best = score
                best_state = {key: value.clone() for key, value in net.state_dict().items()}
        net.load_state_dict(best_state)
        return net, predict

    temporal_net, temporal_predict = fit(lambda: TemporalEncoder(features.shape[-1], HIDDEN, GRU_HIDDEN))
    temporal = temporal_predict(test_starts)
    persistence = baselines.persistence(counts, test_starts, HORIZON)
    history = baselines.historical_mean(
        np.stack([future(s).numpy() for s in train_starts]), len(test_starts)
    )

    window_temporal = evaluate.per_window_mae(actual, temporal)
    window_persistence = evaluate.per_window_mae(actual, persistence)
    beats_baseline = evaluate.paired_bootstrap(window_persistence, window_temporal)
    mask = evaluate.volatile_mask(actual)

    if persist:
        np.savez(
            config.FORECAST_EVAL_PATH,
            coords=coords,
            actual=actual,
            temporal=temporal,
            persistence=persistence,
            volatile=mask,
            cells=np.array(cells),
        )

    metrics = {
        "nodes": len(cells),
        "test_windows": len(test_starts),
        "horizon_days": HORIZON,
        "seed": seed,
        "model": "residual temporal (graph ablated out)",
        "mae": {
            "temporal": evaluate.mae(actual, temporal),
            "persistence": evaluate.mae(actual, persistence),
            "historical_mean": evaluate.mae(actual, history),
        },
        "mae_volatile_cells": {
            "temporal": evaluate.mae(actual[:, mask], temporal[:, mask]),
            "persistence": evaluate.mae(actual[:, mask], persistence[:, mask]),
        },
        "precision_at_50": {
            "temporal": evaluate.precision_at_k(actual, temporal, TOP_K),
            "persistence": evaluate.precision_at_k(actual, persistence, TOP_K),
        },
        "temporal_vs_persistence": {
            "mean_mae_gain": beats_baseline[0],
            "ci95": [beats_baseline[1], beats_baseline[2]],
            "wilcoxon_p": float(wilcoxon(window_persistence - window_temporal).pvalue),
        },
    }

    if with_graph:
        _, spatial_predict = fit(lambda: SpatioTemporalGCN(features.shape[-1], HIDDEN, GRU_HIDDEN))
        spatial = spatial_predict(test_starts)
        window_spatial = evaluate.per_window_mae(actual, spatial)
        graph_effect = evaluate.paired_bootstrap(window_spatial, window_temporal)
        metrics["mae"]["spatial_gnn"] = evaluate.mae(actual, spatial)
        metrics["graph_effect"] = {
            "mean_mae_change": graph_effect[0],
            "ci95": [graph_effect[1], graph_effect[2]],
            "wilcoxon_p": float(wilcoxon(window_spatial - window_temporal).pvalue),
        }

    with torch.no_grad():
        standardised = temporal_net(features_t[len(dates) - LOOKBACK :], edge_index, edge_weight)
        residual = standardised * residual_std + residual_mean
        latest = (counts_t[len(dates) - HORIZON :].sum(0) + residual).clamp(min=0)
    forecast = {cells[i]: float(latest[i]) for i in range(len(cells))}

    if persist:
        config.FORECAST_PATH.write_text(json.dumps({"metrics": metrics, "forecast": forecast}, indent=2))
    return Result(metrics=metrics, forecast=forecast)
