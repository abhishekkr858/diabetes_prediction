import pickle
import streamlit as st
import pandas as pd
import numpy as np

model=pickle.load(open('savemodel.sav','rb'))
scaler=pickle.load(open("scaler.pkl","rb"))

def main():
    st.title('Diabetes Prediction')
    val1=st.text_input('Preg')
    val2=st.text_input('Glucose')
    val3=st.text_input('BloodPressure')
    val4=st.text_input('SkinThickness') 
    val5=st.text_input('Insulin')
    val6=st.text_input('BMI')
    val7=st.text_input('DiabetesPedigreeFunction')
    val8=st.text_input('Age')

    if st.button('Predict'):
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_data = pd.DataFrame([[val1,val2,val3,val4,val5,val6,val7,val8]], 
                          columns=feature_names)
        scaled_values=scaler.transform(input_data)   #[[val1,val2,val3,val4,val5,val6,val7,val8]]
        result=model.predict(scaled_values)
        if result[0]==0:
            result="not diabetic"
        else:
            result="diabetic"
        st.success(f'the patient is {result}')

if __name__=='__main__':
    main()