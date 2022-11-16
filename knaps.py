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

   df = df.drop(columns="id")
   X = df.drop(columns="stroke")
   y = df.stroke

   import Preprocessing
   le = preprocessing.LabelEncoder()
   le.fit(y)
   y = le.transform(y)
   le.inverse_transform(y)
   labels = pd.get_dummies(df.stroke).columns.values.tolist()
   

   dataHasil = pd.concat([df,dataOlah], axis = 1)
   dataHasil
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
   from sklearn.neighbors import KNeighborsClassifier
   from numpy import array
   
   menu = st.sidebar.radio("Pilihan", ['KNN', 'Gaussian Naive Bayes', 'Decision Tree'])

   if test == "KNN":
    st.title("""
        KNN (K-Nearest Neighbor)
        """)
    df = pd.read_csv("https://raw.githubusercontent.com/Rosita19/datamining/main/healthcare-dataset-stroke-data.csv")
    df
    
    metode1 = KNeighborsClassifier(n_neighbors=3)
    metode1.fit(X_train, y_train)
    print(metode1.score(X_train, y_train))
    print(metode1.score(X_test, y_test))
    y_pred = metode1.predict(scaler.transform(array([[50.0,0,1,105.92,0,0,1,0,1,0,1,1,1,1,1,1,1,0,0,0,0]])))
    le.inverse_transform(y_pred)[0]

elif test == "Implementasi":
   st.title("""
      Implementasi Data
      """)



