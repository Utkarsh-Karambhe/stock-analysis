import requests
import pandas as pd

def fetch_stock_data(api_key, symbol):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

def transform_data(data, symbol):
    df = pd.DataFrame(data['Time Series (1min)']).T.reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['symbol'] = symbol
    return df

from sqlalchemy import create_engine

def load_data_to_db(df):
    engine = create_engine('postgresql://uk:gmail1234@localhost:5432/stock_data')
    df.to_sql('stock_prices', engine, if_exists='append', index=False)

if __name__ == "__main__":
    api_key = "MJJTJOTUY0HJKO68"  # Replace with your Alpha Vantage API key
    symbol = "UBER"  # Replace with the stock symbol you want to fetch
    data = fetch_stock_data(api_key, symbol)
    df = transform_data(data, symbol)
    load_data_to_db(df)