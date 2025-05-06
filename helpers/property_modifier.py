import pandas as pd
import unicodedata
import re
from pandas.api.types import is_numeric_dtype
from rapidfuzz import fuzz, process

from helpers.file_manager import FileManager


class PropertyModifier:

    @staticmethod
    def get_unique_dataframe(df, *columns):
        """
        Returns a DataFrame containing only the specified columns with unique rows.
        :param columns: Column names to include.
        :return: DataFrame with unique rows for the given columns.
        """
        if not columns:
            raise ValueError("At least one column must be specified")

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"The following columns are not in the DataFrame: {', '.join(missing_columns)}"
            )

        unique_df = df[list(columns)].drop_duplicates().reset_index(drop=True)
        return unique_df

    @staticmethod
    def add_additional_data(
        df: pd.DataFrame,
        df_key_column: str,
        csv_path: str,
        csv_key_column: str,
        columns_to_exclude: list[str] = [],
        columns_to_include: list[str] = [],
        fuzzy_threshold: int = 85,
        use_fuzzy_matching: bool = False,
        use_partial_matching: bool = False,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        if columns_to_exclude and columns_to_include:
            raise ValueError(
                "Only one of columns_to_exclude or columns_to_include can be specified."
            )

        def normalize_key(value):
            value = str(value)
            # Remove text inside brackets (e.g., "(Pranas)")
            value = re.sub(r"\(.*?\)", "", value)
            # Remove accents
            value = unicodedata.normalize("NFKD", value)
            value = "".join(c for c in value if not unicodedata.combining(c))
            # Remove dots, dashes, etc.
            value = value.replace(".", "").replace("-", " ")
            # Lowercase and sort words alphabetically
            words = value.lower().split()
            # Return a normalized string for fuzzy matching
            return " ".join(sorted(words))

        extra_data = FileManager.read_single_csv(csv_path)

        if columns_to_include:
            # Include only the specified columns and the csv_column
            columns_to_include = set(columns_to_include)
            columns_to_include.add(csv_key_column)
            extra_data = extra_data[
                [col for col in extra_data.columns if col in columns_to_include]
            ]
        else:
            # Exclude the specified columns except the csv_column
            columns_to_exclude = set(columns_to_exclude)
            columns_to_exclude.discard(csv_key_column)
            extra_data = extra_data.drop(
                columns=[col for col in columns_to_exclude if col in extra_data.columns]
            )

        extra_columns = [col for col in extra_data.columns if col != csv_key_column]
        extra_text_cols = []
        extra_numeric_cols = []
        for col in extra_columns:
            if is_numeric_dtype(extra_data[col]):
                extra_numeric_cols.append(col)
            else:
                extra_text_cols.append(col)

        # Normalize keys for joining
        df["match_key"] = df[df_key_column].apply(normalize_key)
        extra_data["match_key"] = extra_data[csv_key_column].apply(normalize_key)

        # Perform exact and partial matching
        matched_keys = set()
        unmatched_keys = set(extra_data["match_key"])
        match_map = {}

        for df_key in df["match_key"]:
            # First, try exact matching
            if df_key in unmatched_keys:
                matched_keys.add(df_key)
                unmatched_keys.discard(df_key)
                match_map[df_key] = df_key
            elif use_partial_matching:
                # Try partial matching if enabled
                for extra_key in unmatched_keys:
                    if set(df_key.split()).issubset(set(extra_key.split())) or set(
                        extra_key.split()
                    ).issubset(set(df_key.split())):
                        matched_keys.add(df_key)
                        unmatched_keys.discard(extra_key)
                        match_map[df_key] = extra_key
                        break
            else:
                # Fuzzy matching as a fallback (only if enabled)
                if use_fuzzy_matching:
                    best_match = process.extractOne(
                        df_key, unmatched_keys, scorer=fuzz.token_sort_ratio
                    )
                    if best_match and best_match[1] >= fuzzy_threshold:
                        matched_keys.add(df_key)
                        unmatched_keys.discard(best_match[0])
                        match_map[df_key] = best_match[0]

        if verbose:
            print(f"Number of connected keys: {len(matched_keys)}")
            print(f"Unmatched keys from '{csv_key_column}': {list(unmatched_keys)}")

        # Map the matched keys back to their original data
        extra_data = extra_data[extra_data["match_key"].isin(matched_keys)]
        extra_data = extra_data.set_index("match_key")
        extra_data = extra_data.rename(
            columns={csv_key_column: f"{csv_key_column}_temp"}
        )
        df = df.set_index("match_key")
        df = df.join(extra_data, how="left")
        df = df.drop(columns=[f"{csv_key_column}_temp"], errors="ignore")
        df = df.reset_index(drop=True)

        return df, extra_text_cols, extra_numeric_cols

    @staticmethod
    def save_missing_data_to_csv(
        df: pd.DataFrame,
        csv_path: str,
        df_key_column: str,
        csv_key_column: str,
        missing_column: str,
        verbose: bool = False,
    ) -> None:
        """
        Appends rows with missing values in the specified column to the CSV file.

        Args:
            df (pd.DataFrame): The DataFrame containing the missing data.
            csv_path (str): The path to the CSV file to update.
            df_key_column (str): The column in the DataFrame used as the key.
            csv_key_column (str): The column in the CSV file used as the key.
            missing_column (str): The column containing the missing values to append.
            verbose (bool): Whether to print detailed logs.
        """
        # Load the existing data from the CSV file
        existing_data = FileManager.read_single_csv(csv_path, separator=";")

        # Filter rows with missing values in the specified column
        missing_data = df[[df_key_column, missing_column]].dropna(
            subset=[missing_column]
        )

        # Rename df_key_column to match csv_key_column for merging
        missing_data = missing_data.rename(columns={df_key_column: csv_key_column})

        # Ensure no duplicates are added by checking against existing data
        merged_data = pd.concat([existing_data, missing_data]).drop_duplicates(
            subset=[csv_key_column]
        )

        # Save the updated data back to the CSV file
        merged_data.to_csv(csv_path, index=False, sep=";")

        if verbose:
            print(f"Appended {len(missing_data)} rows to {csv_path}.")
