import os
import numpy as np
import pandas as pd
from PIL import Image
from data_processing.base_filter import BaseFilter
from datetime import datetime


class InitialCleanupFilter(BaseFilter):
    """
    A filter strategy that performs initial cleanup of the DataFrame.
    It groups transformations by columns or related groups of columns.
    """

    # --- Description & Technique columns ---
    def _clean_description_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Lowercase the 'Description' column.
        df["Description"] = df["Description"].astype(str).str.lower()

        # Derive 'Materials' and 'Surface' from 'Description'
        # The splitting logic depends on the auction name.
        # For rows where the auction is 'Vilniaus Aukcionas', split on comma;
        # otherwise, split on forward slash.
        df["Materials"] = ""
        df["Surface"] = ""
        for idx, row in df.iterrows():
            desc = row["Description"]
            auction_name = row["Auction House"]
            if auction_name == "Vilniaus Aukcionas":
                parts = desc.split(", ") if (isinstance(desc, str) and desc) else []
                if parts:
                    df.at[idx, "Surface"] = parts[0]
                    df.at[idx, "Materials"] = (
                        ", ".join(parts[1:]) if len(parts) > 1 else "nan"
                    )
                else:
                    df.at[idx, "Surface"] = "nan"
                    df.at[idx, "Materials"] = "nan"
            else:
                parts = (
                    desc.split("/")
                    if (isinstance(desc, str) and "/" in desc)
                    else [desc]
                )
                if parts:
                    df.at[idx, "Materials"] = parts[0]
                    df.at[idx, "Surface"] = (
                        "/".join(parts[1:]) if len(parts) > 1 else "nan"
                    )
                else:
                    df.at[idx, "Materials"] = "nan"
                    df.at[idx, "Surface"] = "nan"
        return df

    # --- Price columns ---
    def _clean_price_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Helper functions for price cleaning
        def delete_euro_sign(value):
            value = str(value)
            if "€" in value:
                parts = value.split("€")
                return parts[1].strip() if len(parts) > 1 else value
            return value

        def replace_nan_to_zero(value):
            return 0 if str(value).lower() == "nan" else value

        def replace_not_sold_to_zero(value):
            return 0 if str(value).strip().lower() == "not sold" else value

        def replace_comma(value):
            return str(value).replace(",", "")

        # Process each price column
        for col in [
            "Estimated Minimum Price",
            "Estimated Maximum Price",
            "Sold Price",
            "Primary Price",
        ]:
            df[col] = df[col].apply(delete_euro_sign)
            if col == "Sold Price":
                df[col] = df[col].apply(replace_not_sold_to_zero)
            else:
                df[col] = df[col].apply(replace_nan_to_zero)
            df[col] = df[col].apply(replace_comma)

        # If Estimated Minimum Price is 0, replace it with the cleaned Primary Price value
        mask = df["Estimated Minimum Price"] == 0
        df.loc[mask, "Estimated Minimum Price"] = df.loc[mask, "Primary Price"]

        # Filter out rows where prices are not numeric
        df = df[df["Sold Price"].apply(lambda x: str(x).replace(".", "", 1).isdigit())]
        df = df[
            df["Estimated Minimum Price"].apply(
                lambda x: str(x).replace(".", "", 1).isdigit()
            )
        ]
        return df

    # --- Year columns ---
    def _clean_year_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Helper functions for year cleaning
        def keep_after_symbols(value):
            value = str(value)
            for symbol in ["-", "/", "\\", ","]:
                value = value.split(symbol)[0] if symbol in value else value
            return value

        def delete_unnecessary_sign(value):
            for sign in [
                "act.c.",
                "c.",
                "~",
                "(",
                ")",
                "[",
                "]",
                "'",
                "Iki ",
                "po ",
                ", atrib",
            ]:
                value = str(value).replace(sign, "")
            return value

        def replace_decade(value, decade, year):
            value = str(value)
            return value.replace(decade, year) if decade in value else value

        def replace_question_mark(value):
            return str(value).replace("?", "nan")

        # Process 'Artist Birth Year' and 'Artist Death Year'
        for col in ["Creation Year", "Artist Birth Year", "Artist Death Year"]:
            df[col] = df[col].apply(keep_after_symbols)
            df[col] = df[col].apply(delete_unnecessary_sign)
            df[col] = df[col].apply(lambda x: replace_decade(x, "XXI", "2000"))
            df[col] = df[col].apply(lambda x: replace_decade(x, "XX", "1900"))
            df[col] = df[col].apply(lambda x: replace_decade(x, "XIX", "1820"))
            df[col] = df[col].apply(replace_question_mark)

        # Process 'Auction Date'
        def extract_auction_year(value):
            try:
                parts = str(value).split(" ")
                year = parts[-1]
                return year if year.isnumeric() else "0"
            except Exception:
                return "0"

        df["Auction Date"] = df["Auction Date"].apply(extract_auction_year)

        # If Auction Date includes a range ("to"), take the latter part.
        def leave_years(value):
            value = str(value)
            return value.split("to")[1].strip() if "to" in value else value

        df["Auction Date"] = df["Auction Date"].apply(leave_years)
        return df

    # --- Derived columns based on years and artist info ---
    def _derive_additional_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        current_year = datetime.now().year

        # Derive "Years from auction till now"
        df["Years from auction till now"] = df["Auction Date"].apply(
            lambda x: current_year - int(x) if str(x).isdigit() else "nan"
        )
        # Derive "Years from creation till auction"
        df["Years from creation till auction"] = df.apply(
            lambda row: (
                int(row["Auction Date"]) - int(row["Creation Year"])
                if str(row["Auction Date"]).isdigit()
                and str(row["Creation Year"]).isdigit()
                else "nan"
            ),
            axis=1,
        )
        # Derive "Years from birth till creation"
        df["Years from birth till creation"] = df.apply(
            lambda row: (
                int(row["Creation Year"]) - int(row["Artist Birth Year"])
                if str(row["Creation Year"]).isdigit()
                and str(row["Artist Birth Year"]).isdigit()
                else "nan"
            ),
            axis=1,
        )

        # Determine if the artist is dead
        def is_dead(birth, death):
            if str(death).lower() == "nan":
                if str(birth).isdigit():
                    age = current_year - int(birth)
                    return age > 100
                return "nan"
            return True

        df["Is dead"] = df.apply(
            lambda row: is_dead(row["Artist Birth Year"], row["Artist Death Year"]),
            axis=1,
        )

        # Calculate artist lifetime (or capped lifetime if still alive)
        def artist_lifetime(birth, death):
            if str(death).lower() == "nan":
                if str(birth).isdigit():
                    age = current_year - int(birth)
                    return 100 if age > 100 else 0
                return 0
            else:
                return int(death) - int(birth) if str(birth).isdigit() else 0

        df["Artist Lifetime"] = df.apply(
            lambda row: artist_lifetime(
                row["Artist Birth Year"], row["Artist Death Year"]
            ),
            axis=1,
        )
        # Artist years now (if still alive)
        df["Artist Years Now"] = df.apply(
            lambda row: (
                current_year - int(row["Artist Birth Year"])
                if str(row["Artist Birth Year"]).isdigit()
                and str(row["Artist Death Year"]).lower() == "nan"
                else 0
            ),
            axis=1,
        )

        return df

    def _check_if_singed(self, df: pd.DataFrame) -> pd.DataFrame:
        words_to_find = ["sign.", "signed", "Signed", "Sign."]
        df["Signed"] = ""
        df["Signed"] = df["Details"].apply(
            lambda x: any(word in x for word in words_to_find) if pd.notna(x) else False
        )

        return df

    # --- Dimension columns ---
    def _clean_dimension_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        def replace_comma_to_dot(value):
            return str(value).replace(",", ".")

        df["Width"] = df["Width"].apply(replace_comma_to_dot)
        df["Height"] = df["Height"].apply(replace_comma_to_dot)
        df["Area"] = df.apply(
            lambda row: (
                round(float(row["Width"]) * float(row["Height"]), 2)
                if row["Width"] != "nan" and row["Height"] != "nan"
                else 0
            ),
            axis=1,
        )
        return df

    # --- Calculate aggregated price statistics ---
    def _calculate_min_max_average(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert Sold Price to numeric and drop rows where conversion fails
        df["Sold Price"] = pd.to_numeric(df["Sold Price"], errors="coerce")
        df = df.dropna(subset=["Sold Price"])

        # Compute average, min, and max sold price per artist
        summary = (
            df.groupby("Artist name")["Sold Price"]
            .agg(average_sold_price="mean", min_sold_price="min", max_sold_price="max")
            .reset_index()
        )
        # Round the average price to two decimal places
        summary["average_sold_price"] = summary["average_sold_price"].round(2)

        # Merge the aggregated results back into the original DataFrame
        df = df.merge(summary, on="Artist name", how="left")

        # Rename the columns to the desired names
        df.rename(
            columns={
                "average_sold_price": "Average Sold Price",
                "min_sold_price": "Min Sold Price",
                "max_sold_price": "Max Sold Price",
            },
            inplace=True,
        )
        return df

    def _leave_only_country(self, df: pd.DataFrame) -> pd.DataFrame:
        def _leave_just_country(value):
            value = str(value)
            array = value.split(", ")
            return array[-1]

        df["Auction Country"] = df["Auction City Information"].apply(
            _leave_just_country
        )

        return df

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Process description and technique-related columns.
        df = self._clean_description_columns(df)

        # 2. Clean and standardize price-related columns.
        df = self._clean_price_columns(df)

        # 3. Clean year-related columns.
        df = self._clean_year_columns(df)

        # 4. Derive additional columns based on year and artist data.
        df = self._derive_additional_columns(df)

        # 5. Check if artwork was signed
        df = self._check_if_singed(df)

        # 6. Clean dimensions and calculate artwork area.
        df = self._clean_dimension_columns(df)

        # 7. Calculate aggregated price statistics per artist.
        df = self._calculate_min_max_average(df)

        # 8. Leave only auction country.
        df = self._leave_only_country(df)

        # Drop columns that are no longer needed.
        columns_to_drop = [
            "Auction Date",
            "Auction City Information",
            "Details",
            "Description",
            "Primary Price",
        ]
        df = df.drop(columns=columns_to_drop, errors="ignore")

        return df


class OutlierPriceFilter(BaseFilter):
    """Removes outliers from the 'Sold Price' column using either IQR or Z-score method."""

    def __init__(self, method: str = "iqr", factor: float = 1.5):
        """
        :param method: Outlier detection method. Either "iqr" or "zscore".
        :param factor: Factor to use in outlier detection.
        """
        self.method = method.lower()
        self.factor = factor

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Sold Price" not in df.columns:
            raise KeyError("The DataFrame must contain a 'Sold Price' column.")

        # Convert Sold Price to numeric values
        df_clean = df.copy()
        df_clean["Sold Price"] = df_clean["Sold Price"].apply(self._convert_to_numeric)

        if self.method == "iqr":
            q1 = df_clean["Sold Price"].quantile(0.25)
            q3 = df_clean["Sold Price"].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.factor * iqr
            upper_bound = q3 + self.factor * iqr
            filtered_df = df_clean[
                (df_clean["Sold Price"] >= lower_bound)
                & (df_clean["Sold Price"] <= upper_bound)
            ]
        elif self.method == "zscore":
            mean = df_clean["Sold Price"].mean()
            std = df_clean["Sold Price"].std()
            z_scores = (df_clean["Sold Price"] - mean) / std
            filtered_df = df_clean[abs(z_scores) <= self.factor]
        else:
            raise ValueError(f"Unsupported outlier detection method: {self.method}")

        return filtered_df.reset_index(drop=True)

    def _convert_to_numeric(self, price):
        if pd.isna(price) or (
            isinstance(price, str) and price.strip().lower() == "not sold"
        ):
            return np.nan
        try:
            # Remove euro symbol and commas then convert to float
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
            value = float(str(price).replace("€", "").replace(",", "").strip())
            if value > 0:
                return True
            else:
                return False
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
