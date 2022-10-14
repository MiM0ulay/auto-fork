from app.data_sourcing import Data_Sourcing, data_update
from app.indicator_analysis import Indications
from app.graph import Visualization
from tensorflow.keras.models import load_model
import streamlit as st 
import gc
import pandas as pd
import sys
from datetime import datetime
from binance_assets import BINANCE_ASSETS



gc.collect()
data_update()

# assets = ['ADA', 'AGLD', 'ALPINE', 'ATOM', 'AVAX', 'BAT', 'BNB', 'BTC', 'ETH', 'XRP']
# assets = ['CHESS', '1INCH', 'MULTI', 'STRAX', 'STORJ', 'XLM', 'ARDR']

intervals = {'5m': '5 Minute', '15m': '15 Minute', '30m': '30 Minute',
             '1h': '1 Hour', '6h': '6 Hour', '12h': '12 Hour', '1d': '1 Day', '1w': '1 Week'}
gain_intervals = {'5m': 0.2, '15m': 0.5, '30m': 0.6,
                  '1h': 1.5, '6h': 1.5, '12h': 8.0, '1d': 10.0, '1w': 20.0}


def stat_crypto(app_data, interval, assets, output, gain=0.0):
    st.set_page_config(layout = "wide")
    indication = 'Predicted'
    asset = 'Cryptocurrency'

    if asset in ['Cryptocurrency']:
        exchange = 'Binance'
        app_data.exchange_data(exchange)
        markets = app_data.markets
        market = 'USDT'
        app_data.market_data(market)

        risk = 'Low'  # , 'Medium', 'High'

        df = pd.DataFrame(columns=['Asset',
                                   'Change',
                                   'Cur Price',
                                   'Pre Price',
                                   'Action',
                                   'Buy Price',
                                   'Sell Price',
                                   'CfA',
                                   'CfPr',
                                   'rsi'])
        
        for index, equity in enumerate(assets):
            # print('Computation progress [%d%%]/ %s\r' % (index, len(assets)), end="")
            print('\n' + '=' * 30 + equity + '=' * 30 + '\n')
            try:
                analysis = Visualization(exchange, interval, equity,
                                         indication, action_model, price_model, market)
            except:
                continue
            
            analysis_day = Indications(exchange, '1 Day', equity, market)
            requested_date = analysis.df.index[-1]
            print('**** Requested Price : {}'.format(requested_date))
            current_price = float(analysis.df['Adj Close'][-1])
            current_rsi = float(analysis.df['RSI'][-1])

            if current_price < 0.01:
                continue

            change = float(analysis.df['Adj Close'].pct_change()[-1]) * 100
        
            requested_prediction_price = float(analysis.requested_prediction_price)
            requested_prediction_action = analysis.requested_prediction_action

            risks = {'Low': [analysis_day.df['S1'].values[-1], analysis_day.df['R1'].values[-1]],
                     'Medium': [analysis_day.df['S2'].values[-1], analysis_day.df['R2'].values[-1]],
                     'High': [analysis_day.df['S3'].values[-1], analysis_day.df['R3'].values[-1]], }
            buy_price = float(risks[risk][0])
            sell_price = float(risks[risk][1])

            accuracy_threshold = {analysis.score_action: 75., analysis.score_price: 75.}
            confidence = dict()
            for score, threshold in accuracy_threshold.items():
                if float(score) >= threshold:
                    confidence[score] = f'*({score}%)*'
                else:
                    confidence[score] = ''
        
            potential_gain = ((float(requested_prediction_price) - float(current_price)) / float(current_price)) * 100
            max_gain = ((float(sell_price) - float(buy_price)) / float(buy_price)) * 100

            if max_gain <= gain:
                continue

            new_row = {'Asset': equity,
                       'Change': change,
                       'Cur Price': current_price,
                       'Pre Price': requested_prediction_price,
                       'Action': requested_prediction_action,
                       'Buy Price': buy_price,
                       'Sell Price': sell_price,
                       'potential gain %': potential_gain,
                       'max gain %': max_gain,
                       'CfA': str(confidence[analysis.score_action]),
                       'CfPr': str(confidence[analysis.score_price]),
                       'rsi': current_rsi
                       }
            # append row to the dataframe
            df = df.append(new_row, ignore_index=True)

            df = df.sort_values(by='max gain %', ascending=False)

            df = df.iloc[:20, :]


    st.title(f'Automated Technical Analysis.')
    st.subheader(f'Data Sourced from {exchange}.')
    st.info(f'Predicting...')

    st.dataframe(df, 200, 100)
    


if __name__ == '__main__':
    import warnings
    import gc
    warnings.filterwarnings("ignore") 
    gc.collect()
    action_model = load_model("models/action_prediction_model.h5")
    price_model = load_model("models/price_prediction_model.h5")
    app_data = Data_Sourcing()

    curDT = datetime.now()
    date_time = curDT.strftime("%Y_%m_%d_%H-%M")
    print("date and time:", date_time)

    # Dumping Results
    potential_assets_24h = stat_crypto(
        app_data=app_data, interval='1 Day',
        assets=BINANCE_ASSETS,
        gain=4.0,
        output=f'{date_time}_24h.xlsx')

    potential_assets_12h = stat_crypto(
        app_data=app_data, interval='12 Hour',
        assets=BINANCE_ASSETS,
        gain=3.0,
        output=f'{date_time}_12h.xlsx')

    potential_assets_6h = stat_crypto(
        app_data=app_data, interval='6 Hour',
        assets=potential_assets_24h,
        gain=3.0,
        output=f'{date_time}_6h.xlsx')


