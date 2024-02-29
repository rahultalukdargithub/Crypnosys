
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
import datetime as dt
from keras.models import load_model
import streamlit as st
from yahooquery import Screener
import tickerlist
import re
import plotly.graph_objects as go

from PIL import Image

image_path = "logo.jpg"

import base64

# Define your local image pat

# Read the image file as bytes
image_bytes = open(image_path, "rb").read()

# Convert the image to base64
image_base64 = base64.b64encode(image_bytes).decode()

# Use HTML to align the image in the center
st.markdown(
    f'<div style="display: flex; justify-content: center;">'
    f'<img src="data:image/jpeg;base64,{image_base64}" style="width: 70%;" />'
    f'</div>',
    unsafe_allow_html=True
)

with open('style.css') as f:
  st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
model=load_model('keras_mode_final.h5')

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
    
today=dt.date.today()

st.subheader(f'Data upto {today}')
st.write(df.describe())

st.subheader("Data Visualization")
fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.update_layout(xaxis_rangeslider_visible=False, title=f'Visualize The Data of {user_input}',
                  xaxis_title='Date', yaxis_title='Price')

st.plotly_chart(fig)

st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing Price vs Time chart with 100MA')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
ma100=df.Close.rolling(100).mean()
plt.plot(ma100,'r')
st.pyplot(fig)

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

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

past_200_days=data_training.tail(200)
final_df=past_200_days._append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)



x_test=[]
y_test=[]

for i in range(200,input_data.shape[0]):
  x_test.append(input_data[i-200:i])
  y_test.append(input_data[i,0])


x_test,y_test=np.array(x_test),np.array(y_test)

y_predicted2=model.predict(x_test)
y_predicted2=scaler.inverse_transform(y_predicted2)
y_test=np.array(y_test)
y_test=y_test.reshape(y_test.shape[0],1)
y_test=scaler.inverse_transform(y_test)

st.subheader('Prediction vs Original')
fig=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted2,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


x_test2=[]
x_test2.append(input_data[input_data.shape[0]-200:input_data.shape[0]])
x_test2=np.array(x_test2)


inputt=input_data


x_test2=[]
y_predicted_new=[]

inputtt=input_data

for i in range(7):
  x_test2.append(inputtt[inputtt.shape[0]-200-2:inputtt.shape[0]-2])
  x_test2=np.array(x_test2)
  x_test2=np.reshape(x_test2, (1, x_test2.shape[1], 1))
  y_temp=model.predict(x_test2)
  x_test2 = []
  y_predicted_new.append(y_temp[0])
  # inputtt.append(np.array(y_temp[0]))
  inputtt=np.append(inputtt, [np.array(y_temp[0])], axis=0)



y_predicted_new=np.array(y_predicted_new)


y_predicted_new=scaler.inverse_transform(y_predicted_new)

today = dt.date.today()

dates = [today - dt.timedelta(days=i) for i in range(2, -5, -1)]

st.subheader('Prediction of the Prices in next 5 days')
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
