import os
import numpy as np
import pandas as pd
from PIL import Image
from data_processing.base_filter import BaseFilter


class OutlierPriceFilter(BaseFilter):
    """Removes outliers from the 'Sold Price' column using IQR method."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Sold Price" not in df.columns:
            raise KeyError("The DataFrame must contain a 'Sold Price' column.")

        df_clean = df.copy()
        df_clean["Sold Price"] = df_clean["Sold Price"].apply(self._convert_to_numeric)

        q1 = df_clean["Sold Price"].quantile(0.25)
        q3 = df_clean["Sold Price"].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered_df = df_clean[
            (df_clean["Sold Price"] >= lower_bound)
            & (df_clean["Sold Price"] <= upper_bound)
        ]
        return filtered_df.reset_index(drop=True)

    def _convert_to_numeric(self, price):
        if pd.isna(price) or (
            isinstance(price, str) and price.strip().lower() == "not sold"
        ):
            return np.nan
        try:
            return float(str(price).replace("€", "").replace(",", "").strip())
        except ValueError:
            return np.nan


class InvalidSoldPriceFilter(BaseFilter):
    """Removes rows with invalid or missing sold prices."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Sold Price" not in df.columns:
            raise KeyError("The DataFrame must contain a 'Sold Price' column.")
        return df[df["Sold Price"].apply(self._is_valid_price)]

    def _is_valid_price(self, price) -> bool:
        if pd.isna(price) or (
            isinstance(price, str) and price.strip().lower() == "not sold"
        ):
            return False
        try:
            float(str(price).replace("€", "").replace(",", "").strip())
            return True
        except ValueError:
            return False


class DuplicateFilter(BaseFilter):
    """Removes duplicate rows, ignoring 'Photo id'."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(
            subset=df.columns.difference(["Photo id"]), keep="first"
        ).reset_index(drop=True)


class MissingImageFilter(BaseFilter):
    """Removes rows with missing images."""

    def __init__(self, image_paths):
        self.image_paths = image_paths

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["Photo id"].apply(lambda pid: pid in self.image_paths)]


class LowQualityImageFilter(BaseFilter):
    """Removes low-quality images."""

    def __init__(self, image_paths, min_resolution, min_size_kb):
        self.image_paths = image_paths
        self.min_resolution = min_resolution
        self.min_size_kb = min_size_kb

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["Photo id"].apply(lambda pid: not self._is_low_quality(pid))]

    def _is_low_quality(self, photo_id) -> bool:
        path = self.image_paths.get(photo_id)
        if not path:
            return True

        try:
            size_kb = os.path.getsize(path) / 1024
            if size_kb < self.min_size_kb:
                return True

            with Image.open(path) as img:
                if (
                    img.width < self.min_resolution[0]
                    or img.height < self.min_resolution[1]
                ):
                    return True
        except Exception:
            return True

        return False
