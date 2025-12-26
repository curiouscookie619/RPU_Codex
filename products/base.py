from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, Tuple

from core.models import ParsedPDF, ExtractedFields, ComputedOutputs


class ProductHandler(ABC):
    product_id: str

    @abstractmethod
    def detect(self, parsed: ParsedPDF) -> Tuple[float, Dict[str, Any]]:
        """Return confidence [0,1] and debug info."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, parsed: ParsedPDF) -> ExtractedFields:
        raise NotImplementedError

    @abstractmethod
    def calculate(self, extracted: ExtractedFields, ptd: date) -> ComputedOutputs:
        raise NotImplementedError
