"""Temperature histories for non-isothermal anneals."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemperatureSchedule:
    """Piecewise-linear temperature T(t) in K through (time, temperature) points.

    Held at the first temperature before the first point and at the last one after
    it, so a single point is an isothermal hold. Ramps, spikes and multi-step anneals
    are lists of points, e.g. ``TemperatureSchedule([0, 10, 70, 80], [300, 1300, 1300, 300])``.
    """

    times: tuple[float, ...]
    temperatures: tuple[float, ...]

    def __init__(self, times, temperatures):
        t = tuple(float(v) for v in np.atleast_1d(times))
        temp = tuple(float(v) for v in np.atleast_1d(temperatures))
        if len(t) != len(temp) or not t:
            raise ValueError("times and temperatures need the same, non-zero length")
        if np.any(np.diff(t) <= 0):
            raise ValueError("schedule times must be strictly increasing")
        if min(temp) <= 0:
            raise ValueError("temperatures are absolute and must be positive")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "temperatures", temp)

    @classmethod
    def isothermal(cls, temperature: float) -> TemperatureSchedule:
        return cls([0.0], [temperature])

    def __call__(self, t):
        return np.interp(t, self.times, self.temperatures)


Temperature = float | TemperatureSchedule | Callable[[float], float]


def temperature_at(temperature: Temperature, t: float) -> float:
    """T at time t from a constant, a schedule or any callable of time."""
    return float(temperature(t)) if callable(temperature) else float(temperature)


def with_time(fn: Callable, static_args: int) -> Callable:
    """Return fn as a function of (*static_args, t), whether or not fn takes the time.

    Solver callbacks such as a diffusivity may be written as f(x) for an isothermal
    run or f(x, t) for an anneal; counting the positional parameters tells them apart.
    """
    try:
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
        takes_time = len(params) > static_args or any(p.kind == p.VAR_POSITIONAL for p in inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        takes_time = False
    if takes_time:
        return fn
    return lambda *args: fn(*args[:static_args])
