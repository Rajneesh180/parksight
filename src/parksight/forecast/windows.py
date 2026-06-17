from __future__ import annotations


def window_starts(steps: int, lookback: int, horizon: int) -> list[int]:
    return list(range(lookback, steps - horizon + 1))


def time_split(count: int, train: float = 0.7, val: float = 0.15) -> tuple[list[int], list[int], list[int]]:
    first = int(count * train)
    second = int(count * (train + val))
    return list(range(first)), list(range(first, second)), list(range(second, count))
