import streamlit as st
import re
import pandas as pd 
import numpy as np

st.title("""
PERHITUNGAN DATA
""")

test = st.sidebar.radio("Navigation", ['Home', 'About us', 'Contact us'])

#Fractional Knapsack Problem
#Getting input from user
kodekontrak=int(st.number_input("Kode Kontrak: ",0))
Pendapat=int(st.number_input("Pendapatan Setahun : ",0))
durasipinjaman=int(st.number_input("Durasi Pinjaman : ",0))
jumlahtanggungan=int(st.number_input("Jumlah Tanggungan : ",0))
kpr=str(st.text_input("KPR : ",'ya'))
ovd=str(st.text_input("Rata-Rata Overdue : ",'ya'))

submit = st.button("submit")


if submit:
    st.info("Jadi,dinyakataan . ")


