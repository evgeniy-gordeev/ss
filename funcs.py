import pandas as pd

def transformer_returns(df):
    df = df[['timestamp', 'close']]
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    df = df.sort_values(by = 'timestamp', ascending = True)
    df = df.set_index('timestamp')
    df['return']=df['close'].pct_change()
    df = df.dropna()
    df = df.drop('close', axis = 1)
    return df

def transformer_prices(df):
    df = df[['timestamp', 'close']]
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.date
    df = df.sort_values(by = 'timestamp', ascending = True)
    df = df.set_index('timestamp')
    return df