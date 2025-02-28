from abc import ABC, abstractmethod
import pandas as pd


class BaseFilter(ABC):
    """
    Base class for all data filters in the pipeline.
    Enforces that each filter implements an `apply()` method.
    """

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to the given DataFrame."""
        pass
