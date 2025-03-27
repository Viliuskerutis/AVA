from data_processing.filter_strategies import (
    ArtorkCountFilter,
    EnsureDataFilledAndCorrectFilter,
    DuplicateFilter,
    InitialCleanupFilter,
    InvalidSoldPriceFilter,
    LowQualityImageFilter,
    MissingImageFilter,
    MissingValueFilter,
    OutlierPriceFilter,
    PriceRangeFilter,
)
from data_processing.initial_filter_after_scraping import InitialAfterScrapingFilter


class DataFilterPipeline:
    """A flexible pipeline for applying different filtering strategies."""

    def __init__(self, verbose: bool = True):
        self.steps = []
        self.verbose = verbose

    def add_step(self, step):
        self.steps.append(step)

    def apply(self, df):
        for step in self.steps:
            step_name = step.__class__.__name__  # Get the class name of the step
            initial_length = len(df)

            if self.verbose:
                print(f"Applying {step_name}... (Initial Rows: {initial_length})")
            df = step.apply(df)
            final_length = len(df)

            if self.verbose:
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


def process_after_scraping(df):
    pipeline = DataFilterPipeline()

    pipeline.add_step(InvalidSoldPriceFilter())

    pipeline.add_step(OutlierPriceFilter("iqr", 1.5))
    pipeline.add_step(OutlierPriceFilter("iqr", 1.5))

    pipeline.add_step(InitialAfterScrapingFilter())

    pipeline.add_step(EnsureDataFilledAndCorrectFilter())

    return pipeline.apply(df)


def process_for_predictions(df):
    raise NotImplementedError
    pipeline = DataFilterPipeline()

    pipeline.add_step(InvalidSoldPriceFilter())

    pipeline.add_step(OutlierPriceFilter("iqr", 1.5))
    pipeline.add_step(OutlierPriceFilter("iqr", 1.5))

    pipeline.add_step(InitialCleanupFilter())

    return pipeline.apply(df)


def process_categorical(df):
    pipeline = DataFilterPipeline()

    pipeline.add_step()

    return pipeline.apply(df)


def process_keep_relevant(
    df,
    min_price: float = None,
    max_price: float = None,
    min_artwork_count: int = None,
    max_missing_percent: float = None,
    verbose: bool = True,
):
    pipeline = DataFilterPipeline(verbose)

    if min_price is not None or max_price is not None:
        pipeline.add_step(PriceRangeFilter(min_price, max_price))

    if min_artwork_count is not None:
        pipeline.add_step(ArtorkCountFilter(min_artwork_count))

    if max_missing_percent is not None:
        pipeline.add_step(MissingValueFilter(max_missing_percent))

    return pipeline.apply(df)
