from __future__ import annotations

from typing import Dict, List, Tuple
from .base import ProductHandler
from .gis import GISHandler


_HANDLERS: List[ProductHandler] = [GISHandler()]


def detect_product(parsed) -> Tuple[ProductHandler, float, dict]:
    best = None
    best_conf = -1.0
    best_dbg = {}
    for h in _HANDLERS:
        conf, dbg = h.detect(parsed)
        if conf > best_conf:
            best = h
            best_conf = conf
            best_dbg = dbg
    if best is None:
        raise RuntimeError("No product handlers registered.")
    return best, best_conf, best_dbg
