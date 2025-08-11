"""                     Load the libraries.                     """
import pandas as pd
from typing import Tuple, Optional, List, Any
from sklearn.model_selection import train_test_split


"""                     User defined funtions for data operations.                     """

def describe_dataset(
    X: pd.DataFrame,
    percentiles: Optional[List[float]] = [0.25, 0.5, 0.75],
    include: Any = 'all',
    exclude: Optional[Any] = None
) -> pd.DataFrame:
    """
    Generate a descriptive summary of the dataset, including data types and missing value counts.

    Parameters
    ----------
    X : pandas.DataFrame
        The input DataFrame to describe.
    percentiles : list of float, optional
        List of percentiles to include in the output (default is [0.25, 0.5, 0.75]).
    include : 'all', list-like of dtypes or None, optional
        A white list of data types to include in the result. 'all' includes all columns.
    exclude : list-like of dtypes or None, optional
        A black list of data types to omit from the result.

    Returns
    -------
    df_describe : pandas.DataFrame
        DataFrame with descriptive statistics, data types, and missing value counts.
    """
    df_describe = X.describe(percentiles=percentiles, include=include, exclude=exclude).T.reset_index(names='Feature')
    # Add data type and na_count as second and third columns respectively.
    df_describe.insert(1, 'DataType', X.dtypes.values)
    df_describe.insert(2, 'na_count', X.isna().sum().values)
    return df_describe

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    split_test: float = 0.2,
    split_val: Optional[float] = None,
    random_state: int = 14
) -> Tuple[Any, ...]:
    """
    Split the dataset into training, (optional) validation, and test sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Target vector.
    split_test : float, optional
        Proportion of the dataset to include in the test split (default is 0.2).
    split_val : float or None, optional
        Proportion of the remaining data to include in the validation split (default is None).
        If None, only train and test sets are returned.
    random_state : int, optional
        Random seed for reproducibility (default is 14).

    Returns
    -------
    If split_val is None:
        X_train : pandas.DataFrame
            Training features.
        X_test : pandas.DataFrame
            Test features.
        y_train : pandas.Series
            Training targets.
        y_test : pandas.Series
            Test targets.
    If split_val is not None:
        X_train : pandas.DataFrame
            Training features.
        X_val : pandas.DataFrame
            Validation features.
        X_test : pandas.DataFrame
            Test features.
        y_train : pandas.Series
            Training targets.
        y_val : pandas.Series
            Validation targets.
        y_test : pandas.Series
            Test targets.
    """
    # Split to get the test data.
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=split_test, random_state=random_state, shuffle=True)

    if split_val is None:
        return X_temp, X_test, y_temp, y_test
    else:
        # Split to get train and validation data.
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=split_val, random_state=random_state, shuffle=True)
        return X_train, X_val, X_test, y_train, y_val, y_test
