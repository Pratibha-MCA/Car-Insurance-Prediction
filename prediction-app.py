import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image


html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Car Insurance Prediction App</h2>
    </div>
    """

html_temp2 = """
    <div style="background-color:none;padding:10px">
    <h6 style="text-align:center;">This app predicts Insurance claim of the car</h6>
    </div>
"""

st.markdown(html_temp, unsafe_allow_html=True)
st.markdown(html_temp2, unsafe_allow_html=True)


@st.cache_resource
def pickle_download():
    # Reads in saved classification model
    LR = pickle.load(open('LogisticRegression.pickle', 'rb'))

    return LR

LR = pickle_download()

@st.cache_resource
def download():
    data=pd.read_csv("X.csv")
    
    return data

df_new = download()

@st.cache_resource
def calculation(input_df, df_new):
    # df_new = df_new[]
    SD = pickle.load(open('SD.pickle', 'rb'))
    df = pd.concat([input_df, df_new], axis=0)
    X = SD.transform(df)


    return X[:1]   # Selects only the first row (the user input data)

def turn_off():
    # Delete all the items in Session state
    for key in st.session_state.keys():
        del st.session_state[key]

# Collects user input features into dataframe


def user_input_features():
    KIDSDRIV = int(st.sidebar.selectbox('KIDSDRIV', range(5), on_change=turn_off))
    AGE = float(st.sidebar.slider('AGE', 18, 80, 30, on_change=turn_off))
    HOMEKIDS = float(st.sidebar.selectbox('HOMEKIDS', range(6), on_change=turn_off))
    YOJ = float(st.sidebar.selectbox('YOJ', range(30), on_change=turn_off))
    TIF = float(st.sidebar.selectbox('TIF', range(30), on_change=turn_off))
    CLM_FREQ = float(st.sidebar.selectbox('CLM_FREQ', range(6), on_change=turn_off))
    MVR_PTS = float(st.sidebar.selectbox('MVR_PTS', range(15),  on_change=turn_off))
    CAR_AGE = float(st.sidebar.selectbox('CAR_AGE', range(35), on_change=turn_off))
    Income_converted = float(st.sidebar.slider('Income', 0, 180000, 15000, on_change=turn_off))
    HOME_VAL_converted = float(st.sidebar.slider('HOME_VAL', 0, 500000, 15000, on_change=turn_off))
    BLUEBOOK_converted = float(st.sidebar.slider('BLUEBOOK', 1500, 40000, 10000, on_change=turn_off))
    PARENT1 = (st.sidebar.selectbox('PARENT1', ('YES', 'NO'), on_change=turn_off))
    MSTATUS = (st.sidebar.selectbox('MSTATUS', ('YES', 'NO'),  on_change=turn_off))
    EDUCATION = (st.sidebar.selectbox('EDUCATION', ('Masters', 'High School', 'Bachelors', 'PhD', 'less than High School'), on_change=turn_off))
    CAR_USE = (st.sidebar.selectbox('CAR_USE', ('Private', 'Commercial'), on_change=turn_off))
    CAR_TYPE = (st.sidebar.selectbox('CAR_TYPE', ('SUV', 'Panel Truck', 'Pickup', 'Minivan', 'Sports Car', 'Van'), on_change=turn_off))
    REVOKED = (st.sidebar.selectbox('REVOKED', ('YES', 'NO'), on_change=turn_off))

    LE1 = {'YES':1, 'NO':0}
    LE2 = {'YES':0, 'NO':1}
    LE3 = {'Masters':2, 'High School':4, 'Bachelors':1, 'PhD':3, 'less than High School':0}
    LE4 = {'Private':1, 'Commercial':0}
    LE5 = {'SUV':5, 'Panel Truck':1, 'Pickup':2, 'Minivan':0, 'Sports Car':3, 'Van':4}
    PARENT1 = LE1[PARENT1]
    MSTATUS = LE2[MSTATUS]
    REVOKED = LE1[REVOKED]
    EDUCATION = LE3[EDUCATION]
    CAR_USE = LE4[CAR_USE]
    CAR_TYPE = LE5[CAR_TYPE]



    data = {'KIDSDRIV': KIDSDRIV,
            'AGE': AGE,
            'HOMEKIDS': HOMEKIDS,
            'YOJ': YOJ,
            'TIF': TIF,
            'CLM_FREQ': CLM_FREQ,
            'MVR_PTS': MVR_PTS,
            'CAR_AGE': CAR_AGE,
            'Income_converted': Income_converted,
            'HOME_VAL_converted': HOME_VAL_converted,
            'BLUEBOOK_converted': BLUEBOOK_converted,
            'LE_PARENT1': PARENT1,
            'LE_MSTATUS': MSTATUS,
            'LE_EDUCATION': EDUCATION,
            'LE_CAR_USE': CAR_USE,
            'LE_CAR_TYPE': CAR_TYPE,
            'LE_REVOKED': REVOKED,
            }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()



safe_html="""  
      <div style="background-color:#8CE57A;padding:10px >
       <h2 style="color:white;text-align:center;">NO</h2>
       </div>
    """
danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;">YES</h2>
       </div>
    """
if "button1" not in st.session_state:
    st.session_state.button1 = False

def callback():
    st.session_state.button1 = True

button1 = st.button("Will there be a claim?",on_click=callback)

if button1 or st.session_state.button1:

    df = calculation(input_df, df_new)

    # Apply model to make predictions
    prediction = LR.predict(df)

    st.subheader('Prediction')
    if prediction == 1:
        st.markdown(danger_html, unsafe_allow_html=True)
    else:
        st.markdown(safe_html, unsafe_allow_html=True)
    st.write(' ')


