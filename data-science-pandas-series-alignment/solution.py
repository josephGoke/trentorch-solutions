import numpy as np
import pandas as pd


def make_series(values, labels, name) -> pd.Series:
    """Returns a Series of `values` indexed by `labels` (same order), named `name`."""
    series = pd.Series(data=values, index=labels, name=name)
    return series
    


def aligned_sum(a: pd.Series, b: pd.Series, fill_value=None) -> pd.Series:
    """
    Adds a and b by label. The index is every label from either side, sorted
    ascending. A label missing from one side gives NaN. With fill_value, a
    missing label or NaN value on either side counts as fill_value, unless
    both sides are missing for that label (then NaN). Does not change a or b.
    """
    a.sort_index()
    b.sort_index()
    if fill_value is None:
        total = a + b
    else:
        total = a.add(b, fill_value=fill_value)
    return total.sort_index()        
    


def share_of_total(s: pd.Series) -> pd.Series:
    """
    Each value divided by the sum of the non-missing values. Missing values
    stay missing. Same index and name as s, float values.
    """
    # TODO: Divide by the total of the known values.
    s = s.astype(float)
    return s / s.sum(skipna=True)


def lookup(s: pd.Series, labels) -> pd.Series:
    """
    Value of s for each label in `labels`, in that order, indexed by `labels`.
    A label that s does not have gives NaN instead of an error.
    """
    return s.reindex(list(labels))
