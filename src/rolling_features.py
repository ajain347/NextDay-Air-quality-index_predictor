"""Week 3 — Ishan: past-only rolling pollutant features."""
import pandas as pd


def add_rolling_features(df: pd.DataFrame, pollutant: str = 'PM2.5') -> pd.DataFrame:
    """Create 3/7-day past-only rolling mean/std on continuous Delhi daily rows.

    Input must have a Date column, unique consecutive daily dates, and a
    numeric pollutant column. Missing observations are retained: windows
    containing missing values yield NaN rather than silently imputing.
    """
    if 'Date' not in df.columns or pollutant not in df.columns:
        raise KeyError(f"Expected 'Date' and '{pollutant}' columns")
    result = df.copy()
    result['Date'] = pd.to_datetime(result['Date'], errors='raise')
    if result['Date'].isna().any():
        raise ValueError('Date has missing values')
    result = result.sort_values('Date').reset_index(drop=True)
    if result['Date'].duplicated().any():
        raise ValueError('Duplicate dates; resolve before rolling features')
    if not result['Date'].diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
        raise ValueError('Dates must be continuous daily observations before rolling')
    past = pd.to_numeric(result[pollutant], errors='raise').shift(1)
    safe_name = pollutant.replace('.', '').replace(' ', '_')
    for window in (3, 7):
        result[f'{safe_name}_roll_mean_{window}'] = past.rolling(window, min_periods=window).mean()
        result[f'{safe_name}_roll_std_{window}'] = past.rolling(window, min_periods=window).std()
    return result
