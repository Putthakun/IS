import streamlit as st
import joblib
import pandas as pd
import numpy as np

# โหลดโมเดลสุนัข
model_dog = joblib.load('dog_breed_model.pkl')
label_encoders_dog = joblib.load('label_encoders.pkl')
label_encoder_breed = label_encoders_dog['breed']
label_encoder_traits = label_encoders_dog['traits']

# โหลดโมเดลดอกไม้
flower_model_path = r"C:\Users\BIG GER\Desktop\IS\IS\model2"
rf_name = joblib.load(f"{flower_model_path}\\rf_name.pkl")
rf_perfumes = joblib.load(f"{flower_model_path}\\rf_perfumes.pkl")
rf_color = joblib.load(f"{flower_model_path}\\rf_color.pkl")
label_encoders_flower = joblib.load(f"{flower_model_path}\\label_encoders.pkl")
le_name = label_encoders_flower['le_name']  # สมมติว่าเก็บใน dict นี้
le_perfumes = label_encoders_flower['le_perfumes']
mlb = label_encoders_flower['mlb']  # สมมติว่าเก็บ MultiLabelBinarizer

# เมนูเลือกหน้า
page = st.selectbox("Select a page", ["Main", "Dog Breed Machine", "Flower Predictor", "Neural Network", "Machine Learning"])

if page == "Main":
    # หน้า Main (Welcome Page)
    st.title("🌟 Intelligent System 🌟")
    st.write("""
        🤖 **Hello and Welcome!**  
        This is a basic knowledge center about intelligent systems. With basic AI models, please come and try it.! 🐶✨
    """)

elif page == "Dog Breed Machine":
    # หน้า Dog Breed Machine
    st.title("Dog Breed Character Traits Predictor")
    st.write("Enter a dog breed to predict its character traits.")

    st.subheader("Dataset Information")
    st.write("""
        This dataset is sourced from Kaggle. It includes the following features:
        - **Country of Origin**: The country where the breed originated.
        - **Breed**: The breed of the dog.
        - **Fur Color**: The typical color of the dog’s fur.
        - **Height (inches)**: The height of the dog in inches.
        - **Color of Eyes**: The typical eye color of the breed.
        - **Longevity (years)**: The average lifespan of the breed.
        - **Character Traits**: The predicted character traits of the breed.
        - **Common Health Problems**: Common health issues related to the breed.
    """)

    breed_input = st.text_input("Enter a dog breed (Labrador Retriever,German Shepherd,Bulldog,Poodle(พิมพ์ใหญ่ตัวเเรกเสมอ)):", "").strip()

    if st.button("Predict"):
        if not breed_input:
            st.error("Please enter a breed.")
        else:
            if breed_input not in label_encoder_breed.classes_:
                st.warning(f"Breed '{breed_input}' Try again.")
            else:
                try:
                    breed_encoded = label_encoder_breed.transform([breed_input]).reshape(1, -1)
                    traits_encoded = model_dog.predict(breed_encoded)
                    traits = label_encoder_traits.inverse_transform(traits_encoded)[0]
                    st.success(f"Predicted Character Traits for {breed_input}: {traits}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "Flower Predictor":
    # หน้า Flower Predictor
    st.title("Flower Predictor")
    st.write("Enter flower height and longevity to predict its name, perfume presence, and colors.")

    st.subheader("Dataset Information")
    st.write("""
        This dataset includes the following features:
        - **Height (cm)**: The height of the flower in centimeters.
        - **Longevity (years)**: The average lifespan of the flower.
        - **Name**: The name of the flower.
        - **Perfumes**: Whether the flower has a perfume scent.
        - **Color**: The typical colors of the flower.
    """)

    height = st.number_input("Enter flower height (cm):", min_value=0.0, step=0.1)
    longevity = st.number_input("Enter flower longevity (years):", min_value=0.0, step=0.1)

    if st.button("Predict Flower"):
        if height <= 0 or longevity <= 0:
            st.error("Please enter valid numeric values greater than 0.")
        else:
            try:
                input_data = pd.DataFrame([[height, longevity]], columns=['height (cm)', 'longevity (years)'])
                
                name_pred = le_name.inverse_transform(rf_name.predict(input_data))[0]
                perfumes_pred = le_perfumes.inverse_transform(rf_perfumes.predict(input_data))[0]
                color_pred = mlb.inverse_transform(rf_color.predict(input_data))
                
                result_message = f"""
                ===== Prediction Result =====  
                🌸 **Name**: {name_pred}  
                🌿 **Has Perfume**: {'Yes' if perfumes_pred else 'No'}  
                🎨 **Colors**: {', '.join(color_pred[0]) if color_pred else 'Unknown'}
                """
                st.success(result_message)
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif page == "Neural Network":
    # หน้า Neural Network
    st.title("Neural Network")
    st.write("A Neural Network is a type of machine learning model inspired by the way the human brain works. It is made up of layers of interconnected nodes (also called neurons), which process information in a way similar to how neurons in the brain transmit signals.")
    st.write("Here's a breakdown of its components:")
    st.write("Input Layer: This is where the neural network receives input data. Each node in this layer represents a feature of the input data.")
    st.write("Hidden Layers: These are layers between the input and output layers, where the actual computation happens.")
    st.write("Output Layer: This layer produces the final result or prediction based on the inputs and computations of the hidden layers.")

elif page == "Machine Learning":
    # หน้า Machine Learning
    st.title("Machine Learning")
    st.write("Machine Learning (ML) is a field of artificial intelligence (AI) that focuses on developing algorithms and models that enable computers to learn from data and make decisions or predictions without being explicitly programmed.")
    st.write("In traditional programming, the programmer writes a set of rules or instructions to tell the computer what to do. In machine learning, the computer learns from examples in the data.")