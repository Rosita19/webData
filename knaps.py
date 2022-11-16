import streamlit as st
import re
import pandas as pd 
import numpy as np

st.title("""
PERHITUNGAN DATA
""")

test = st.sidebar.radio("Menu", ['Learn Data', 'Preprocessing', 'Model', 'Implementasi'])

if test == "Learn Data":
    st.subheader("Hai, saya Jumadi :wave:")
    st.title("Saya Seorang coach artificial intelligence dari Indonesia")
    st.write(
        "Saya bersemang untuk menggunakan Streamlit lebih efisien dan efektif dalam bisnis."
    )
    st.write("[Pelajari Lebih Lanjut](https://kelasawanpintar.netlify.app/)")

    st.write("---")

    st.header("Apa yang saya lakukan")
    st.write("##")
    st.write(
            """
            Di Channel YouTube saya, saya membuat tutorial untuk orang-orang yang:
            - sedang mencari cara untuk belajar Python.
            - sedang mencari cara untuk belajar Streamlit.
            - ingin belajar Analisis Data & Ilmu Data untuk melakukan analisis.
            - ingin belajar Artificial Intelligence, Data Science, Machine Learning, Natural Language Processing.
            - ingin belajar dunia IT
            - Jika ingin terhubung di [Linkedin](https://www.linkedin.com/in/jumadi-01/)


            Jika Channel YouTube saya menarik bagi Anda, jangan lupa untuk berlangganan dan menyalakan notifikasi, agar Anda tidak ketinggalan konten apa pun.
            
            [Channel YouTube](https://www.youtube.com/channel/UC7rCdlKnMTt26Q3np3rW1Iw)
            """
        )
    st.header("Playlist Fundamental Streamlit")
    col1, col2, col3 = st.columns(3)

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


