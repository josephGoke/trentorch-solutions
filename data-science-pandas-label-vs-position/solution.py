import pandas as pd


def rows_by_label(df: pd.DataFrame, start, stop) -> pd.DataFrame:
    """Rows from label `start` to label `stop`, both included, in frame order."""
    return df.loc[start:stop]


def rows_by_position(df: pd.DataFrame, start: int, stop: int) -> pd.DataFrame:
    """Rows at positions start..stop-1 (stop excluded), in frame order."""
    return df.iloc[start:stop]


def cell(df: pd.DataFrame, row_label, column):
    """The single value at row label `row_label` and column name `column`."""
    value = df.loc[row_label, column]
    return value

def numeric_columns(df: pd.DataFrame) -> list:
    """Names of the integer and float columns (not bool), in column order."""
    cols = df.select_dtypes(include=[int,float]).columns
    return cols.tolist()


def with_value_where(df: pd.DataFrame, mask, column, value) -> pd.DataFrame:
    """
    A new frame where `column` is `value` on the rows where boolean `mask` is
    true. The input frame is not changed.
    """
    frame = df.copy()

    frame.loc[mask, column] = value

    return frame
