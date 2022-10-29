import pandas as pd
import streamlit as st
from processing import func as process

dataset = pd.read_csv('Wine_Quality.csv')
st.title("Results from sample data:")
df=process(dataset)
st.table(df.head())
st.line_chart(df)

st.title("Try on custom data")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataset=pd.read_csv(uploaded_file)
    df=process(dataset)
    st.line_chart(df)