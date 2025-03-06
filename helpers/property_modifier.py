import pandas as pd
import unicodedata

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
        df_column: str,
        csv_path: str,
        csv_column: str,
        columns_to_exclude: list[str] = [],
    ):
        def normalize_name(value):
            value = str(value)

            # Remove accents
            value = unicodedata.normalize("NFKD", value)
            value = "".join(c for c in value if not unicodedata.combining(c))

            # Remove dots, dashes, etc.
            value = value.replace(".", "").replace("-", " ")

            # Lowercase and split into words
            words = value.lower().split()

            # Sort words alphabetically and rejoin
            return " ".join(sorted(words))

        extra_data = FileManager.read_single_csv(csv_path)

        columns_to_exclude = set(columns_to_exclude)  # Handle duplicates
        columns_to_exclude.discard(
            csv_column
        )  # Always keep the match column for merging

        extra_data = extra_data.drop(
            columns=[col for col in columns_to_exclude if col in extra_data.columns]
        )

        df["match_key"] = df[df_column].apply(normalize_name)
        extra_data["match_key"] = extra_data[csv_column].apply(normalize_name)

        extra_data = extra_data.drop(columns=[csv_column])
        extra_data = extra_data.set_index("match_key")
        df = df.set_index("match_key")
        df = df.join(extra_data, how="left")
        df = df.reset_index(drop=True)

        return df
