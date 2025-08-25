import pandas as pd

def next_day_return_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['Target'] = out['Return'].shift(-1)
    return out