import streamlit as st
import pandas as pd

st.write("# My First Streamlit App")
st.write("Here's our first attempt at using data to create a table:")
data = pd.DataFrame ({
    'Column 1': [1, 2, 3, 4],
    'Column 2': ['A', 'B', 'C', 'D']
})
st.write(data)
