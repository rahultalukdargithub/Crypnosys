
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
import datetime as dt
# from keras.models import load_model
import streamlit as st
import tickerlist
import re
import plotly.graph_objects as go
import requests
import time
import pandas_ta as ta
import plotly.express as px
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
image_path = "logo.png"

import base64


image_bytes = open(image_path, "rb").read()


image_base64 = base64.b64encode(image_bytes).decode()

st.markdown(
    f'<div style="display: flex; justify-content: center;">'
    f'<img src="data:image/jpeg;base64,{image_base64}" style="width: 70%;" />'
    f'</div>',
    unsafe_allow_html=True
)

with open('style.css') as f:
  st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
# model=load_model('keras_mode_final.h5')

# Function to get data with caching
def get_data(user_input, start, end):
    df = pdr.get_data_yahoo(user_input, start, end)
    return df

def extract_content_within_brackets(s):
    match = re.search(r'\((.*?)\)', s)
    return match.group(1) if match else None

# s = Screener()
# data = s.get_screeners('all_cryptocurrencies_us', count=250)
# dicts = data['all_cryptocurrencies_us']['quotes']
# symbols = [d['symbol'] for d in dicts]
# symbols_tuple = tuple(symbols)

yfin.pdr_override()
start='2013-02-15'
end=dt.datetime.now()
st.markdown("<div id='title'>_Ultimate Predictor_</div>", unsafe_allow_html=True)
st.text('')
st.text('')

asset_type = st.selectbox('Select From The Below Shown List :', ['Cryptocurrency', 'Stock','Index','ETF','Mutual Fund'])

if asset_type == 'Cryptocurrency':
    Crypto_Ticker = tickerlist.Crypto_Ticker
    selected_crypto = st.selectbox('Select Cryptocurrency Ticker:', Crypto_Ticker)
    df = get_data(extract_content_within_brackets(selected_crypto), start, end)
    user_input = selected_crypto
elif asset_type == 'Stock':
    stock_ticker=tickerlist.stock_ticker
    selected_stock = st.selectbox('Select Stock Ticker:', stock_ticker)
    df = get_data(extract_content_within_brackets(selected_stock), start, end)
    user_input = selected_stock
elif asset_type == 'ETF':
    ETF_Ticker=tickerlist.ETF_Ticker
    selected_ETF = st.selectbox('Select ETF Ticker:', ETF_Ticker)
    df = get_data(extract_content_within_brackets(selected_ETF), start, end) 
    user_input = selected_ETF
elif asset_type == 'Index':
    Index_Ticker=tickerlist.Index_Ticker
    selected_Index = st.selectbox('Select Index Ticker:', Index_Ticker)
    df = get_data(extract_content_within_brackets(selected_Index), start, end)    
    user_input = selected_Index
else:
    Mutual_Ticker=tickerlist.Mutual_Ticker    
    selected_Mutual = st.selectbox('Select Mutual Fund Ticker:', Mutual_Ticker)
    df = get_data(extract_content_within_brackets(selected_Mutual), start, end) 
    user_input = selected_Mutual
    


if(df.shape[0]>=200):
    today=dt.date.today()
    st.subheader(f'Data upto {today}')
    st.write(df.describe())
    dfs=df.tail(200)
    dfs=dfs.reset_index()
    dfs = dfs.set_index(pd.DatetimeIndex(dfs['Date'].values))
    st.subheader("Data Visualization")
    fig = go.Figure(data=[go.Candlestick(x=dfs.index,
                    open=dfs['Open'],
                    high=dfs['High'],
                    low=dfs['Low'],
                    close=dfs['Close'],
                    increasing_line_color = 'green',
                    decreasing_line_color='red')])
    fig.update_layout(title=f'Visualize The Data of {user_input}',
                    xaxis_title='Date', yaxis_title='Price',width=860,height=600,margin=dict(l=50, r=50, t=50, b=50) )

    st.plotly_chart(fig)

    st.subheader('Closing Price vs Time chart')
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)



    # st.subheader('Closing Price vs Time chart with 100MA')
    # fig=plt.figure(figsize=(12,6))
    # plt.plot(df.Close)
    # ma100=df.Close.rolling(100).mean()
    # plt.plot(ma100,'r')
    # st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA and 200MA')
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    st.pyplot(fig)

    data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
    data_testing=pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])


    # scaler=MinMaxScaler(feature_range=(0,1))
    scaler=RobustScaler()

    data_training_array = scaler.fit_transform(data_training)

    x_train=[]
    y_train=[]


    for i in range(200,data_training_array.shape[0]):
        x_train.append(data_training_array[i-200:i])
        y_train.append(data_training_array[i,0])


    x_train,y_train = np.array(x_train),np.array(y_train)

    model = xgb.XGBRegressor(n_estimators=170,learning_rate=0.08, max_depth=3)
    x_train = x_train.reshape(x_train.shape[0], -1)
    model.fit(x_train, y_train)


    past_200_days=data_training.tail(200)
    final_df=past_200_days._append(data_testing,ignore_index=True)
    input_data=scaler.fit_transform(final_df)



    col1, col2 = st.columns(2)
    if(asset_type == "Cryptocurrency"):
        with col1:
            timeFrame = st.selectbox('Select The Time-Frame:', ["60", "3600", "86400"])
        with col2:
            nbars = st.selectbox('Select number of bars:', ["20", "50", "100"])
    else:
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        with col1:
            per = st.selectbox('Select The Period :', valid_periods)
        with col2:  
            interva = st.selectbox('Select The Interval :', valid_intervals)  
        
        



    st.subheader('Stream The Live Data')
    placeholder=st.empty()
    while True:
            with placeholder.container():

                try:
                    if(asset_type == "Cryptocurrency"):
                        original_string = extract_content_within_brackets(user_input)
                        modified_string = original_string.lower().replace("-", "")
                        
                        url =f"https://www.bitstamp.net/api/v2/ohlc/{modified_string}/"
                        params={
                            "step":timeFrame,
                            "limit":int(nbars)+14,
                        }
                        data=requests.get(url,params=params).json()["data"]["ohlc"]
                        data = pd.DataFrame(data)
                        data.timestamp=pd.to_datetime(data.timestamp,unit="s")
                        data["rsi"]= ta.rsi(data.close.astype(float))
                        data=data.iloc[14:]
                        
                        fig = go.Figure(data=[go.Candlestick(x=data.timestamp,
                                    open=data.open,
                                    high=data.high,
                                    low=data.low,
                                    close=data.close,
                                    )])
                        fig.update_layout(title=f'Recent Price of {user_input}',
                            xaxis_title='Date', yaxis_title='Price',width=890,xaxis_rangeslider_visible=False,height=500,margin=dict(l=50, r=50, t=50, b=50) )            
                        indicator =px.line(x=data.timestamp,y=data.rsi,height =200,width=860)
                        time.sleep(0.25)  
                        st.plotly_chart(fig)
                        st.plotly_chart(indicator)
                    else:
                        try:
                            data = yfin.download(tickers=extract_content_within_brackets(user_input), period=per, interval=interva,rounding=True,auto_adjust=True)
                            data = pd.DataFrame(data)
                            data=data.reset_index()
                            data.Datetime = pd.to_datetime(data['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                            data["rsi"]= ta.rsi(data.Close.astype(float))
                            data=data.tail(14)
                            
                            fig = go.Figure(data=[go.Candlestick(x=data.Datetime,
                                                    open=data.Open,
                                                    high=data.High,
                                                    low=data.Low,
                                                    close=data.Close,
                                                    )])
                            fig.update_layout(title=f'Recent Price of {user_input}',
                                            xaxis_title='Date', yaxis_title='Price',width=890,xaxis_rangeslider_visible=False,height=500,margin=dict(l=50, r=50, t=50, b=50) ) 
                                    
                            indicator =px.line(x=data.Datetime,y=data.rsi,height =200,width=860)   
                            time.sleep(0.25)  
                            st.plotly_chart(fig)
                            st.plotly_chart(indicator)
                        except Exception as e:
                            st.write("Try again with different value of Period and Interval , if its not working then we will Soon Add It ")
                except Exception as e: 
                    st.write("We Will Soon Add It")
                    
                        
                y_predicted_new=[]
                x_test2=[]
                inputtt=input_data

                for i in range(7):
                    x_test2.append(inputtt[inputtt.shape[0]-200-2:inputtt.shape[0]-2])
                    x_test2=np.array(x_test2)
                    x_test2=np.reshape(x_test2, (1, x_test2.shape[1]))
                    # x_test2 = x_test2.reshape(x_test.shape[0], -1)
                    y_temp=model.predict(x_test2)
                    x_test2 = []
                    y_predicted_new.append(y_temp)
                    # inputtt.append(np.array(y_temp[0]))
                    inputtt=np.append(inputtt, [np.array(y_temp)], axis=0)

                y_predicted_new=np.array(y_predicted_new)

                y_predicted_new =y_predicted_new.reshape(-1, 1)

                y_predicted_new=scaler.inverse_transform(y_predicted_new)

                today = dt.date.today()

                dates = [today - dt.timedelta(days=i) for i in range(2, -5, -1)]

                st.subheader(f'Prediction of the prices of {user_input} in next 5 days')
                fig=plt.figure(figsize=(12, 6))
                plt.plot(dates, y_predicted_new, 'r', marker='o', label='Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig)
                if(y_predicted_new[2][0] > y_predicted_new[6][0]):
                    st.write(f'Crypnosys predicts these range of Price {y_predicted_new[6][0]} - {y_predicted_new[2][0]} in {user_input} for the next 5 days')
                elif(y_predicted_new[2][0] < y_predicted_new[6][0]):
                    st.write(f'Crypnosys predicts these range of Price {y_predicted_new[2][0]} - {y_predicted_new[6][0]} in {user_input} for the next 5 days')
                else:
                    st.write(f'Crypnosys predicts the Price will be same - {y_predicted_new[6][0]} in {user_input} for the next 5 days')
                st.markdown("<div id='custom-divider'></div>", unsafe_allow_html=True)
                st.caption("These results have been generated by a machine and may differ from actual values. Use them at your own risk.")
                
                image_url = "https://cdn-icons-png.flaticon.com/256/3670/3670209.png"
                
                # Link URL
                link_url = "https://www.youtube.com/channel/UCDaKB-foogVsUBI3yyQXzag"
                
                # Image size
                image_width = 45
                image_height = 45
                
                # Display image with embedded link
                st.markdown(
                    f"<div style='display: flex; justify-content: center;'><a href='{link_url}' target='_blank'><img src='{image_url}' width='{image_width}' height='{image_height}'></a></div>",
                    unsafe_allow_html=True
                )
else:
    st.write("Currently, information for this particular asset is not available, but rest assured, we are in the process of adding it soon. In the meantime, feel free to peruse our diverse range of other available assets.")


