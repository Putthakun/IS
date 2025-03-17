import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# โหลดโมเดลสุนัข
model = joblib.load('dog_breed_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
label_encoder_breed = label_encoders['breed']
label_encoder_traits = label_encoders['traits']

# โหลดโมเดลดอกไม้
rf_name = joblib.load('rf_name.pkl')
rf_perfumes = joblib.load('rf_perfumes.pkl')
rf_color = joblib.load('rf_color.pkl')
le_name = joblib.load('le_name.pkl')
le_perfumes = joblib.load('le_perfumes.pkl')
mlb = joblib.load('mlb.pkl')

# เมนูเลือกหน้า
page = st.selectbox("Select a page", ["Main", "Dog Breed Machine", "Neural Network", "Machine Learning", "Flower Predictor"])

if page == "Main":
    st.title("🌟 Intelligent System 🌟")
    st.write("""
        🤖 **Hello and Welcome!**  
        This is a basic knowledge center about intelligent systems. With basic AI models, please come and try it.! 🐶✨
    """)

elif page == "Dog Breed Machine":
    st.title("Dog Breed Character Traits Predictor")
    breed_input = st.text_input("Enter a dog breed:", "").strip()
    if st.button("Predict"):
        if not breed_input:
            st.error("Please enter a breed.")
        else:
            if breed_input not in label_encoder_breed.classes_:
                st.warning(f"Breed '{breed_input}' not found. Try again.")
            else:
                breed_encoded = label_encoder_breed.transform([breed_input]).reshape(1, -1)
                traits_encoded = model.predict(breed_encoded)
                traits = label_encoder_traits.inverse_transform(traits_encoded)[0]
                st.success(f"Predicted Character Traits for {breed_input}: {traits}")

elif page == "Neural Network":
    st.title("Neural Network")
    st.write("A brief introduction to Neural Networks.")

elif page == "Machine Learning":
    st.title("Machine Learning")
    st.write("A brief introduction to Machine Learning.")

elif page == "Flower Predictor":
    st.title("Flower Predictor")
    st.write("Predict flower name, perfumes, and colors based on height and longevity.")
    
    height = st.number_input("Enter flower height (cm):", min_value=0.0, format="%.2f")
    longevity = st.number_input("Enter flower longevity (years):", min_value=0.0, format="%.2f")
    
    if st.button("Predict Flower"):
        input_data = pd.DataFrame([[height, longevity]], columns=["height (cm)", "longevity (years)"])
        name_pred = le_name.inverse_transform(rf_name.predict(input_data))[0]
        perfumes_pred = le_perfumes.inverse_transform(rf_perfumes.predict(input_data))[0]
        color_pred = mlb.inverse_transform(rf_color.predict(input_data))
        
        st.write(f"**Predicted Flower Name:** {name_pred}")
        st.write(f"**Has Perfume:** {'Yes' if perfumes_pred else 'No'}")
        st.write(f"**Colors:** {', '.join(color_pred[0]) if color_pred else 'Unknown'}")
