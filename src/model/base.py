from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModel(ABC):
    @abstractmethod
    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        """Defines the forward pass of the model."""
        pass

    @abstractmethod
    def get_position_ids(self, **kwargs: Any) -> Any:
        """Generates position IDs for the given inputs."""
        pass

    @abstractmethod
    def shift_position_ids(self, **kwargs: Any) -> Any:
        """Shifts position IDs for autoregressive generation."""
        pass
