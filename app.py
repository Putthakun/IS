import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load('dog_breed_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
label_encoder_breed = label_encoders['breed']
label_encoder_traits = label_encoders['traits']

# เมนูเลือกหน้า
page = st.selectbox("Select a page", ["Main", "Dog Breed Machine", "Neural Network", "Machine Learning"])

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

    # ข้อมูลเพิ่มเติมเกี่ยวกับ Dataset
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

    # ช่องป้อนข้อมูล
    breed_input = st.text_input("Enter a dog breed (Labrador Retriever,German Shepherd,Bulldog,Poodle(พิมพ์ใหญ่ตัวเเรกเสมอ)):", "").strip()

    # ปุ่มทำนาย
    if st.button("Predict"):
        if not breed_input:
            st.error("Please enter a breed.")
        else:
            if breed_input not in label_encoder_breed.classes_:
                st.warning(f"Breed '{breed_input}' Try again.")
            else:
                try:
                    breed_encoded = label_encoder_breed.transform([breed_input]).reshape(1, -1)
                    traits_encoded = model.predict(breed_encoded)
                    traits = label_encoder_traits.inverse_transform(traits_encoded)[0]
                    st.success(f"Predicted Character Traits for {breed_input}: {traits}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

elif page == "Neural Network":
    # หน้า Neural Network
    st.title("Neural Network")
    st.write("A Neural Network is a type of machine learning model inspired by the way the human brain works. It is made up of layers of interconnected nodes (also called neurons), which process information in a way similar to how neurons in the brain transmit signals.")
    st.write("Here's a breakdown of its components:")
    st.write("Input Layer: This is where the neural network receives input data. Each node in this layer represents a feature of the input data.")
    st.write("Hidden Layers: These are layers between the input and output layers, where the actual computation happens. These layers consist of neurons that apply weights and biases to the inputs, followed by an activation function to determine the output of each neuron.")
    st.write("Output Layer: This layer produces the final result or prediction based on the inputs and computations of the hidden layers.")
elif page == "Machine Learning":
    # หน้า Machine Learning
    st.title("Machine Learning")
    st.write("Machine Learning (ML) is a field of artificial intelligence (AI) that focuses on developing algorithms and models that enable computers to learn from data and make decisions or predictions without being explicitly programmed.")
    st.write("In traditional programming, the programmer writes a set of rules or instructions to tell the computer what to do. In machine learning, instead of explicitly programming the rules, the computer learns from examples in the data, recognizing patterns and making predictions based on them.")