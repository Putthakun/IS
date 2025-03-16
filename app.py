import streamlit as st
import joblib
import numpy as np

# โหลดไฟล์ CSS จากโฟลเดอร์ templates
st.markdown('<link href="templates/styles.css" rel="stylesheet">', unsafe_allow_html=True)

# โหลดโมเดล
model = joblib.load('dog_breed_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
label_encoder_breed = label_encoders['breed']
label_encoder_traits = label_encoders['traits']

# เมนูเลือกหน้า
page = st.selectbox("Select a page", ["Home", "Neural Network", "Machine Learning"])

if page == "Home":
    # หน้า Home
    st.title("Dog Breed Character Traits Predictor")
    st.write("ข้อมูลเกี่ยวกับการเตรียมข้อมูลและทฤษฎีการพัฒนาโมเดล")

    # ข้อมูลเกี่ยวกับ Dataset
    st.subheader("ข้อมูลเกี่ยวกับ Dataset")
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
    breed_input = st.text_input("Enter a dog breed (e.g., Labrador):", "").strip()

    # ปุ่มทำนาย
    if st.button("Predict"):
        if not breed_input:
            st.error("Please enter a breed.")
        else:
            if breed_input not in label_encoder_breed.classes_:
                st.warning(f"Breed '{breed_input}' not found in the dataset.")
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
    st.write("""
        A Neural Network is a type of machine learning model inspired by the way biological neural networks in the human brain process information. 
        It's designed to recognize patterns, learn from data, and make predictions or decisions.

        A neural network consists of layers of nodes, also known as neurons, which are connected to each other in a network. The main components of a neural network are:

        - **Input Layer**: The first layer, which takes in the raw data (e.g., an image, text, or numerical values).
        - **Hidden Layers**: Layers between the input and output, where computations and transformations happen. These layers help the network learn complex patterns in the data.
        - **Output Layer**: The final layer, which produces the network's prediction or decision.
    """)

elif page == "Machine Learning":
    # หน้า Machine Learning
    st.title("Machine Learning")
    st.write("""
        Machine Learning (ML) is a field of artificial intelligence (AI) that focuses on developing algorithms and models that enable computers to learn from data and make decisions or predictions without being explicitly programmed.

        There are three main types of machine learning:

        - **Supervised Learning**: The model is trained on labeled data (input with correct output). The goal is for the model to learn the relationship between inputs and outputs and make predictions for new data.
        - **Unsupervised Learning**: The model is trained on unlabeled data, seeking hidden patterns or groupings within the data.
        - **Reinforcement Learning**: An agent learns by interacting with the environment and receiving feedback through rewards or punishments.
    """)
