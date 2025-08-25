import pandas as pd

def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def add_basic_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'Close': close})
    df['Return'] = df['Close'].pct_change()
    for k in [1,2,3,5,10]:
        df[f'Lag{k}'] = df['Return'].shift(k)
    df['Vol_5'] = df['Return'].rolling(5).std()
    df['EMA_10'] = ema(df['Close'], 10)
    df['EMA_20'] = ema(df['Close'], 20)
    df['EMA_gap'] = (df['EMA_10'] - df['EMA_20']) / df['EMA_20']
    return df