import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ตรวจสอบการนำเข้า TensorFlow
try:
    import tensorflow as tf
    
except ImportError as e:

    raise

# โหลดโมเดลสุนัข
model_dog = joblib.load('dog_breed_model.pkl')
label_encoders_dog = joblib.load('label_encoders.pkl')
label_encoder_breed = label_encoders_dog['breed']
label_encoder_traits = label_encoders_dog['traits']

# โหลดโมเดลดอกไม้จากโฟลเดอร์ model2
flower_model_path = "model2"
rf_name = joblib.load(f"{flower_model_path}/rf_name.pkl")
rf_perfumes = joblib.load(f"{flower_model_path}/rf_perfumes.pkl")
rf_color = joblib.load(f"{flower_model_path}/rf_color.pkl")
label_encoders_flower = joblib.load(f"{flower_model_path}/label_encoders.pkl")
le_name = label_encoders_flower['name']
le_perfumes = label_encoders_flower['perfumes']
mlb = label_encoders_flower['color']

# โหลดโมเดลไวน์จากโฟลเดอร์ model2
try:
    wine_model = tf.keras.models.load_model(f"{flower_model_path}/wine_quality_model.h5")

    scaler_wine = joblib.load(f"{flower_model_path}/scaler.pkl")
except Exception as e:
    st.error(f"Failed to load Neural Wine Quality model or scaler: {str(e)}")
    raise

# เมนูเลือกหน้า
page = st.selectbox("Select a page", ["Main", "Dog Breed Machine", "Flower Predictor", "Neural Wine Quality Predictor", "Neural Network", "Machine Learning"])

if page == "Main":
    st.title("🌟 Intelligent System 🌟")
    st.write("""
        🤖 **Hello and Welcome!**  
        This is a basic knowledge center about intelligent systems. With basic AI models, please come and try it.! 🐶✨
    """)
    st.write("Thank you for providing the dataset on Kaggle. It's concise and very helpful for my project.")

elif page == "Dog Breed Machine":
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

elif page == "Neural Wine Quality Predictor":
    st.title("Neural Wine Quality Predictor")
    st.write("Enter the following features to predict the quality of wine using a neural network model.")

    st.subheader("Dataset Information")
    st.write("""
        This dataset includes the following features:
        - **Fixed Acidity**: Fixed acidity level in the wine.
        - **Volatile Acidity**: Volatile acidity level in the wine.
        - **Citric Acid**: Citric acid content in the wine.
        - **Residual Sugar**: Amount of residual sugar in the wine.
        - **Chlorides**: Chloride content in the wine.
        - **Free Sulfur Dioxide**: Free sulfur dioxide content in the wine.
        - **Total Sulfur Dioxide**: Total sulfur dioxide content in the wine.
        - **Density**: Density of the wine.
        - **pH**: pH level of the wine.
        - **Sulphates**: Sulphate content in the wine.
        - **Alcohol**: Alcohol percentage in the wine.
        - **Quality**: The predicted quality of the wine (output).
    """)

    selected_features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
    ]
    
    features = {}
    for feature in selected_features:
        features[feature] = st.number_input(f"Enter {feature}:", min_value=0.0, step=0.01)

    if st.button("Predict Neural Wine Quality"):
        if any(value <= 0 for value in features.values()):
            st.error("Please enter valid numeric values greater than 0 for all features.")
        else:
            try:
                input_data = pd.DataFrame([list(features.values())], columns=selected_features)
                input_data_scaled = scaler_wine.transform(input_data)
                prediction = wine_model.predict(input_data_scaled)
                st.success(f"Predicted Wine Quality (Neural Model): {prediction[0][0]:.2f}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

elif page == "Neural Network":
    st.title("Neural Network")
    st.write("A Neural Network is a type of machine learning model inspired by the structure and functioning of the human brain. It’s designed to recognize patterns in data.")
    
    st.write("Here’s a breakdown of its key components:")
    
    st.write("#### 1. Input Layer:")
    st.write("The input layer receives the raw data, which can be images, text, numbers, etc. Each node in this layer corresponds to a feature or attribute of the input data.")
    
    st.write("#### 2. Hidden Layers:")
    st.write("Hidden layers are intermediate layers that process and transform the input data using weighted connections. Each layer’s output is fed into the next layer, gradually refining the information.")
    
    st.write("#### 3. Output Layer:")
    st.write("The output layer generates the final predictions or classifications based on the processed data from the hidden layers. For example, in a classification task, the output layer could output the probability of each class.")
    
    st.write("#### How it Works:")
    st.write("A neural network learns by adjusting the weights of the connections between neurons based on the error in its predictions. This process is called **training** and is done using algorithms like backpropagation.")
    
    st.write("Neural networks are powerful tools used in many applications, such as image recognition, natural language processing, and even playing games like chess or Go!")

elif page == "Machine Learning":
    st.title("Machine Learning")
    st.write("Machine Learning (ML) is a field of artificial intelligence (AI) that focuses on developing algorithms and models.")
    st.write("In traditional programming, the programmer writes a set of rules. In machine learning, the computer learns from examples.")
    st.write("There are several types of machine learning:")
    st.write("Supervised Learning: The model is trained on labeled data, meaning the input data is paired with the correct output. It learns to map inputs to the correct outputs.")
    st.write("Unsupervised Learning: The model  The model is given data without explicit labels and must find patterns and structure in the data, such as grouping similar data points together.")
    st.write("Reinforcement Learning: The model learns by interacting with an environment and receiving feedback through rewards or penalties, improving over time based on its actions.")