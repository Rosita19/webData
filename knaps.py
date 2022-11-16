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
elif test == "Preprocessing":
   st.title("""
      Pemrosesan Data
      """)
   df = pd.read_csv("https://raw.githubusercontent.com/Rosita19/datamining/main/healthcare-dataset-stroke-data.csv")
   df
   kodekontrak=int(st.number_input("Kode Kontrak: ",0))
   Pendapat=int(st.number_input("Pendapatan Setahun : ",0))
   durasipinjaman=int(st.number_input("Durasi Pinjaman : ",0))
   jumlahtanggungan=int(st.number_input("Jumlah Tanggungan : ",0))
   kpr=str(st.text_input("KPR : ",'ya'))
   ovd=str(st.text_input("Rata-Rata Overdue : ",'ya'))

   submit = st.button("submit")
   
   if submit:
       st.info("Jadi,dinyakataan . ")

elif test == "Model":
   st.title("""
      Modeling 
      """)

elif test == "Implementasi":
   st.title("""
      Implementasi Data
      """)



