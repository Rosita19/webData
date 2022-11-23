import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

Home, Learn, Proses, Model, Implementasi = st.tabs(['Home', 'Learn Data', 'Preprocessing', 'Model', 'Implementasi'])

with Home:
   st.title("""PENAMBANGAN DATA C""")
   st.write('Rosita Dewi Lutfiyah/200411100002')

with Learn:
   st.title("""Data Prediksi Gagal Jantung""")
   st.write('Penyakit kardiovaskular (CVDs) adalah penyebab kematian nomor 1 secara global, merenggut sekitar 17,9 juta nyawa setiap tahun, yang merupakan 31% dari semua kematian di seluruh dunia. Empat dari kematian 5CVD disebabkan oleh serangan jantung dan stroke, dan sepertiga dari kematian ini terjadi sebelum waktunya pada orang di bawah usia 70 tahun. Gagal jantung adalah kejadian umum yang disebabkan oleh CVD dan kumpulan data ini berisi 11 fitur yang dapat digunakan untuk memprediksi kemungkinan penyakit jantung.')
   st.write('Dengan penyakit kardiovaskular atau yang memiliki risiko kardiovaskular tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia, atau penyakit yang sudah ada) memerlukan deteksi dan penanganan dini di mana model pembelajaran mesin dapat sangat membantu.')
   st.title('Atribut Informasi')
   st.write('1. Umur : umur pasien [tahun]')
   st.write('2. Kelamin: jenis kelamin pasien [L: Pria, F: Wanita]')
   st.write('3. Tipe Nyeri Dada: tipe nyeri dada [TA: Angina Khas, ATA: Angina Atipikal, NAP: Nyeri Non-Angina, ASY: Asimtomatik]')
   st.write('4. Istirahat BP: tekanan darah istirahat [mm Hg]')
   st.write('5. Kolesterol: kolesterol serum [mm/dl]')
   st.write('6. FastingBS: gula darah puasa [1: jika FastingBS > 120 mg/dl, 0: jika tidak]')
   st.write('7. EKG istirahat: hasil elektrokardiogram istirahat [Normal: Normal, ST: memiliki kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST > 0,05 mV), LVH: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes]')
   st.write('8. MaxHR: detak jantung maksimum tercapai [Nilai numerik antara 60 dan 202]')
   st.write('9. ExerciseAngina: angina akibat olahraga [Y: Ya, N: Tidak]')
   st.write('10. Oldpeak: oldpeak = ST [Nilai numerik diukur dalam depresi]')
   st.write('11. ST_Slope: kemiringan segmen ST latihan puncak [Up: upsloping, Flat: flat, Down: downsloping]')
   st.write('12. HeartDisease: kelas keluaran [1: penyakit jantung, 0: Normal]')

   st.title("""Upload Data""")
   st.write("Dataset yang digunakan adalah Heart Failure Prediction dataset yang diambil dari https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction")
   st.write("Total datanya adalah 918 dengan atribut 12")
   uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
   for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with Proses:
   st.title("""Preprosessing""")
   df[["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]].agg(['min','max'])

   df = df.drop(columns=["FastingBS", "RestingECG"])
   X = df.drop(columns="HeartDisease")
   y = df.HeartDisease
   "### Membuang fitur yang tidak diperlukan"
   df

   le = preprocessing.LabelEncoder()
   le.fit(y)
   y = le.transform(y)

   "### Transformasi Label"
   y

   le.inverse_transform(y)

   labels = pd.get_dummies(df.HeartDisease).columns.values.tolist()

   "### Label"
   labels

   "### Normalize data"

   dataubah=df.drop(columns=['Sex','ChestPainType','ExerciseAngina','ST_Slope'])
   dataubah

   
   "### Normalize data Sex"
   data_sex=df[['Sex']]
   sex = pd.get_dummies(data_sex)
   sex

   "### Normalize data ChestPainType"
   data_ces=df[['ChestPainType']]
   ces = pd.get_dummies(data_ces)
   ces

   "### Normalize data ExerciseAngina"
   data_ex=df[['ExerciseAngina']]
   ex = pd.get_dummies(data_ex)
   ex

   "### Normalize data ST_Slope"
   data_slop=df[['ST_Slope']]
   slop = pd.get_dummies(data_slop)
   slop

   dataOlah = pd.concat([sex,ces,ex,slop], axis=1)
   dataHasil = pd.concat([df,dataOlah], axis = 1)

   X = dataHasil.drop(columns=['Sex','ChestPainType','ExerciseAngina','ST_Slope'])
   y = dataHasil.HeartDisease
   "### Normalize data hasil"
   X

   scaler = MinMaxScaler()
   scaler.fit(X)
   X = scaler.transform(X)
   "### Normalize data transformasi"
   X

   X.shape, y.shape

   le.inverse_transform(y)

   labels = pd.get_dummies(dataHasil.HeartDisease).columns.values.tolist()
    
   "### Label"
   labels

   scaler = MinMaxScaler()
   scaler.fit(X)
   X = scaler.transform(X)
   X

   X.shape, y.shape

with Model:
   st.title("""Modeling""")
   X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
   from sklearn.preprocessing import StandardScaler
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   # from sklearn.feature_extraction.text import CountVectorizer
   # cv = CountVectorizer()
   # X_train = cv.fit_transform(X_train)
   # X_test = cv.fit_transform(X_test)
   st.subheader("Ada beberapa pilihan model dibawah ini!")
   st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
   naive = st.checkbox('Naive Bayes')
   kn = st.checkbox('K-Nearest Neighbor')
   des = st.checkbox('Decision Tree')
   mod = st.button("Modeling")

   # NB
   GaussianNB(priors=None)

   # Fitting Naive Bayes Classification to the Training set with linear kernel
   nvklasifikasi = GaussianNB()
   nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

   # Predicting the Test set results
   y_pred = nvklasifikasi.predict(X_test)
    
   y_compare = np.vstack((y_test,y_pred)).T
   nvklasifikasi.predict_proba(X_test)
   akurasi = round(100 * accuracy_score(y_test, y_pred))
   # akurasi = 10

   # KNN 
   K=10
   knn=KNeighborsClassifier(n_neighbors=K)
   knn.fit(X_train,y_train)
   y_pred=knn.predict(X_test)

   skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

   # DT

   dt = DecisionTreeClassifier()
   dt.fit(X_train, y_train)
   # prediction
   dt.score(X_test, y_test)
   y_pred = dt.predict(X_test)
   #Accuracy
   akurasiii = round(100 * accuracy_score(y_test,y_pred))

   if naive :
      if mod == 'Modeling' :
         st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
   if knn :
      if mod == '':
         st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
   if dt :
      if mod :
         st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
   eval = st.button("Evaluasi semua model")
   if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)


with Implementasi:
   st.title("""Implementasi Data""")
   Age = st.number_input('Masukkan Umur Pasien')

   # Sex
   sex = st.radio("Sex",('Male', 'Female'))
   if sex == "Male":
      gen_Female = 0
      gen_Male = 1
   elif sex == "Female" :
      gen_Female = 1
      gen_Male = 0

   # ChestPainType
   Ces = st.radio("ChestPainType",('ATA', 'NAP', 'ASY'))
   if Ces == "ATA":
      ces_ata = 1
      ces_nap = 0
      ces_asy = 0
   elif Ces == "NAP":
      ces_ata = 0
      ces_nap = 1
      ces_asy = 0
   elif Ces == "ASY":
      ces_ata = 0
      ces_nap = 0
      ces_asy = 1
   
   Res = st.number_input('Masukkan Resting BP')
   col = st.number_input('Masukkan kolesterol')

   # FastingBS
   fas = st.radio("FastingBs",('No', 'Yes'))
   if fas == "Yes":
      fas = 1
   elif fas == "No":
      fas = 0

   # Resting ECG
   res_ecg = st.radio("RestingECG",('Normal', 'ST'))
   if res_ecg == "Normal":
      ecgN = 1
      ecgS = 0
   elif res_ecg == "ST":
      ecgN = 0
      ecgS = 1

   maks = st.number_input('Masukkan MaxHR')

   # Exercise Angina
   ex = st.radio("ExerciseAngina",('N', 'Y'))
   if ex == "N":
      ex_n = 1
      ex_y = 0
   elif ex == "Y":
      ex_n = 0
      ex_y = 1

   old = st.number_input('Masukkan Oldpeak')

   # ST Slope
   slop = st.radio("ST_Slope",('Up', 'Flat'))
   if slop == "Up":
      slop_u = 1
      slop_f = 0
   elif slop == "Flat":
      slop_u = 0
      slop_f = 1

   # HeartDisease
   heart = st.radio("HeartDisease",('No', 'Yes'))
   if heart == "Yes":
      heart = 1
   elif heart == "No":
      heart = 0

   def submit():
      # input
      inputs = np.array([[
         Age,
         gen_Female, gen_Male,
         ces_ata, ces_nap, ces_asy,
         Res, col, fas,
         ecgN, ecgS,
         maks, ex_n, ex_y,
         old, slop_u, slop_f, heart
         ]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
      le = joblib.load("le.save")
      model1 = joblib.load("knn.joblib")
      y_pred3 = model1.predict(inputs)
      st.write(f"Berdasarkan data yang Anda masukkan, maka anda dinyatakan : {le.inverse_transform(y_pred3)[0]}")

   all = st.button("Submit")
   if all :
        st.balloons()
        submit()



