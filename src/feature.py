import pandas as pd


def add_features(df):
    """
    Create engineered features used by the house price model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the original house features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the engineered features added.
    """

    df = df.copy()

    # Total basement + ground floor + second floor area
    df["TotalSF"] = (
        df["TotalBsmtSF"]
        + df["1stFlrSF"]
        + df["2ndFlrSF"]
    )

    # Total bathrooms, counting half bathrooms as 0.5
    df["TotalBathrooms"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )

    # Total porch and deck area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["3SsnPorch"]
        + df["EnclosedPorch"]
        + df["ScreenPorch"]
        + df["WoodDeckSF"]
    )

    # Age of the house when it was sold
    df["HouseAge"] = (
        df["YrSold"] - df["YearBuilt"]
    )

    # Years since the last remodeling when sold
    df["RemodAge"] = (
        df["YrSold"] - df["YearRemodAdd"]
    )

    # Total finished basement area
    df["TotalBsmtFinSF"] = (
        df["BsmtFinSF1"]
        + df["BsmtFinSF2"]
    )

    return df