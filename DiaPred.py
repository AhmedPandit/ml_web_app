import numpy as np
import pickle 
import streamlit as st


loaded_model=pickle.load(open("trained_model.sav",'rb'))

# Creating a function for prediction

def diabetes_prediction(input_data):
    input_data_np=np.asanyarray(input_data)
    input_data_reshaped=input_data_np.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    
    if(prediction[0]==0):
        return 'This person is not diabetic'
    else:
        return 'This person is  diabetic'


def main():
    # giving a title for user page
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from user
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose')
    BloodPressure=st.text_input('BP')
    SkinThickness=st.text_input('skin thickness')
    Insulin=st.text_input('Insulin')
    BMI =st.text_input('BMI')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree')
    Age=st.text_input('Age')
    
    diagnosis=''
    
    if st.button("Diabetes Test Result"):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
if __name__=='__main__':
    main()