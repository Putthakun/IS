import joblib

# โหลดไฟล์
try:
    label_encoders_flower = joblib.load("model2/label_encoders.pkl")
    print("Contents of label_encoders_flower:", label_encoders_flower)

    # ถ้าเป็น dictionary ให้พิมพ์คีย์ทั้งหมด
    if isinstance(label_encoders_flower, dict):
        print("Keys in label_encoders_flower:", list(label_encoders_flower.keys()))
    else:
        print("label_encoders_flower is not a dictionary. Type:", type(label_encoders_flower))
except Exception as e:
    print("Error loading label_encoders.pkl:", str(e))