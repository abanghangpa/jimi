"""Abstract base class for all scoring modules."""

from abc import ABC, abstractmethod


class BaseModule(ABC):
    """Every scoring module must implement score()."""

    @abstractmethod
    def score(self, data, direction, **kwargs):
        """
        Score the module for a given direction.

        Args:
            data: DataFrame or dict with required data
            direction: 'LONG' or 'SHORT'
            **kwargs: module-specific parameters

        Returns:
            tuple: (status: str, score: float, details: dict)
            status: 'PASS', 'FAIL', 'SKIP', or 'NEUTRAL'
            score: 0.0 to 1.0
            details: dict of diagnostic info
        """
        pass
