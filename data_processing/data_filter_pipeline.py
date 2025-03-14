from data_processing.filter_strategies import (
    DuplicateFilter,
    InitialCleanupFilter,
    InvalidSoldPriceFilter,
    LowQualityImageFilter,
    MissingImageFilter,
    OutlierPriceFilter,
    PriceRangeFilter,
)


class DataFilterPipeline:
    """A flexible pipeline for applying different filtering strategies."""

    def __init__(
        self,
    ):
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def apply(self, df):
        for step in self.steps:
            step_name = step.__class__.__name__  # Get the class name of the step
            initial_length = len(df)

            print(f"Applying {step_name}... (Initial Rows: {initial_length})")
            df = step.apply(df)
            final_length = len(df)

            print(
                f"Completed {step_name}. (Remaining Rows: {final_length}, Removed: {initial_length - final_length})"
            )

        return df


def process_for_image_similarity(df, image_path_dictionary):
    pipeline = DataFilterPipeline()

    pipeline.add_step(InvalidSoldPriceFilter())

    pipeline.add_step(MissingImageFilter(image_path_dictionary))
    pipeline.add_step(DuplicateFilter())
    pipeline.add_step(
        LowQualityImageFilter(
            image_path_dictionary, min_resolution=(250, 250), min_size_kb=20
        )
    )

    return pipeline.apply(df)


def process_for_predictions(df):
    pipeline = DataFilterPipeline()

    pipeline.add_step(InvalidSoldPriceFilter())

    pipeline.add_step(InitialCleanupFilter())

    pipeline.add_step(
        PriceRangeFilter(min_price=None, max_price=None)
    )  # Keep at `None` for no filtering

    pipeline.add_step(OutlierPriceFilter("iqr", 1.5))

    return pipeline.apply(df)
