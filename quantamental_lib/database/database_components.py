import numpy as np
from datetime import timedelta, date
import yfinance as yf
import requests
import pandas as pd
import json

class DatabaseComponents:
    def __init__(self) -> None:
        pass

    def _fetch_returns(self, data, open_date, close_date):
        ticker = (data.columns[0]).split('_')[0]
        data = data.dropna()
        daily_retuns = (data-data.shift(1))/data.shift(1)
        daily_retuns = daily_retuns.fillna(0)
        df = pd.DataFrame(daily_retuns)
        df = df.rename(columns={ticker+'_close': 'RET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        return df

    def _fetch_log_returns(self, data, open_date, close_date):
        ticker = (data.columns[0]).split('_')[0]
        data = data.dropna()
        daily_retuns = np.log((data)/data.shift(1))
        daily_retuns = daily_retuns.fillna(0)
        df = pd.DataFrame(daily_retuns)
        df = df.rename(columns={ticker+'_close': 'LRET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        return df

    def _fetch_cumulated_returns(self, data, open_date, close_date):
        ticker = (data.columns[0]).split('_')[0]
        data = data.dropna()
        daily_retuns = (data-data.shift(1))/data.shift(1)
        cumulated_returns = (daily_retuns.fillna(0)+1).cumprod()-1
        df = pd.DataFrame(cumulated_returns)
        df = df.rename(columns={ticker+'_close': 'CRET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        return df

    def _fetch_cumulated_log_returns(self, data, open_date, close_date):
        ticker = (data.columns[0]).split('_')[0]
        data = data.dropna()
        daily_retuns = np.log((data)/data.shift(1))
        cumulated_returns = (daily_retuns.fillna(0)).cumsum()
        df = pd.DataFrame(cumulated_returns)
        df = df.rename(columns={ticker+'_close': 'CLRET_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        return df

    def _fetch_volatility(self, data, periods, open_date, close_date):
        ticker = (data.columns[0]).split('_')[0]
        data = data.dropna()
        daily_retuns = (data-data.shift(1))/data.shift(1)
        volatility = (daily_retuns.rolling(periods).std())*(periods**(1/2))
        df = pd.DataFrame(volatility)
        df = df.rename(columns={ticker+'_close': 'VOL' +
                        str(periods)+'_'+ticker+'_close'})
        df = df.loc[open_date:close_date]
        return df
    
    def get_brazilian_tickers(self):
        url_request = ("https://brapi.dev/api/available")
        rqst = requests.get(url_request)
        obj = json.loads(rqst.text)
        error = obj.get('error')
        if error:
            return False
        data = obj['stocks']
        return data
    
    def get_most_traded(self, 
                    maximum_date=None, 
                    previous_days_to_consider: int = 30, 
                    number_of_tickers: int = 100):
        if maximum_date is None:
            maximum_date = pd.to_datetime(date.today())
        open_date = maximum_date - timedelta(days=previous_days_to_consider)
        br_tickers_raw = self.get_brazilian_tickers()
        br_tickers = [ticker + ".SA" for ticker in br_tickers_raw]
        df = yf.download(br_tickers, start=open_date,
                        end=maximum_date, progress=False)
        volume_info = dict(df['Volume'].sum())
        ordered_volume = dict(
            sorted(volume_info.items(), key=lambda item: item[1], reverse=True))
        tickers_raw = list(ordered_volume.keys())[:number_of_tickers]
        tickers = [ticker.replace(".SA", "") for ticker in tickers_raw]
        return tickers



