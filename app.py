
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin
import datetime as dt
from keras.models import load_model
import streamlit as st

from PIL import Image

img = Image.open("logo1.jpg")
st.image(
  img,
)

with open('style.css') as f:
  st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
model=load_model('keras_mode_final.h5')

yfin.pdr_override()
start='2013-02-15'
end=dt.datetime.now()
user_input=st.text_input('Enter The Crypto name or Stock Ticker:','BTC-USD')
df = pdr.get_data_yahoo(user_input, start, end)


today=dt.date.today()

st.subheader(f'Data upto {today}')
st.write(df.describe())


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

st.subheader('prediction of the prices in next 5 days')
fig=plt.figure(figsize=(12, 6))
plt.plot(dates, y_predicted_new, 'r', marker='o', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


if(y_predicted_new[2][0] > y_predicted_new[6][0]):
    st.subheader(f'Crypnosys predicts these range of Price {y_predicted_new[6][0]} - {y_predicted_new[2][0]} in {user_input} for the next 5 days')
elif(y_predicted_new[2][0] < y_predicted_new[6][0]):
    st.subheader(f'Crypnosys predicts these range of Price {y_predicted_new[2][0]} - {y_predicted_new[6][0]} in {user_input} for the next 5 days')
else:
    st.subheader(f'Crypnosys predicts the Price will be same - {y_predicted_new[6][0]} in {user_input} for the next 5 days')
