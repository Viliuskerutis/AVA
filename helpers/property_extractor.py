class PropertyExtractor:

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
