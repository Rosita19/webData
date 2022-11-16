import streamlit as st
import re
import pandas as pd 
import numpy as np

test = st.sidebar.radio("Menu", ['Learn Data', 'Preprocessing', 'Model', 'Implementasi'])

if test == "Learn Data":
   st.title("""
      Data Pasien Menderita Stroke
      """)
   df = pd.read_csv("https://raw.githubusercontent.com/Rosita19/datamining/main/healthcare-dataset-stroke-data.csv")
   df


