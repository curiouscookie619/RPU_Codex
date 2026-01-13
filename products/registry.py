from __future__ import annotations

from typing import Dict, List, Tuple
from .base import ProductHandler
from .gis import GISHandler
from .fsp import FSPHandler


class ProductNotConfigured(Exception):
    """Raised when the uploaded BI does not match any configured product."""


_HANDLERS: List[ProductHandler] = [GISHandler(), FSPHandler()]


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
    if best_conf <= 0:
        raise ProductNotConfigured("Product not configured yet; RPU calculation is not available.")
    return best, best_conf, best_dbg
