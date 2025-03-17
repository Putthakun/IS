import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# โหลดข้อมูล
file_path = "/content/flowers.csv"  # อัปโหลดไฟล์ไปยัง Google Colab แล้วใช้ path นี้
df = pd.read_csv(file_path)

# ----- เตรียมข้อมูล -----
# ฟังก์ชันแปลงช่วงตัวเลขเป็นค่าเฉลี่ย
def convert_range_to_mean(value):
    try:
        return np.mean([int(i) for i in str(value).split('-')])
    except ValueError:
        return np.nan  # ถ้าค่าไม่สามารถแปลงเป็นตัวเลขได้ ให้เป็น NaN แทน

# จัดการข้อมูลที่มีปัญหา
for col in ['height (cm)', 'longevity (years)']:
    df[col] = df[col].replace("Variable", np.nan)  # แทนที่ 'Variable' ด้วย NaN
    df[col] = df[col].apply(convert_range_to_mean)
    df = df.dropna(subset=[col])  # ลบแถวที่มี NaN

# แปลง target `name` เป็นตัวเลข
le_name = LabelEncoder()
df['name'] = le_name.fit_transform(df['name'])

# แปลง target `perfumes` เป็น binary
le_perfumes = LabelEncoder()
df['perfumes'] = le_perfumes.fit_transform(df['perfumes'])

# แปลง `color` ให้เป็น Multi-label binary
mlb = MultiLabelBinarizer()
df['color'] = df['color'].apply(lambda x: x.split(', '))
color_labels = mlb.fit_transform(df['color'])
color_df = pd.DataFrame(color_labels, columns=mlb.classes_)
df = df.drop(columns=['color']).join(color_df)

# เลือก features และ target
X = df[['height (cm)', 'longevity (years)']]
y_name = df['name']
y_perfumes = df['perfumes']
y_color = color_labels  # Multi-label

# แบ่งข้อมูล train/test
X_train, X_test, y_name_train, y_name_test = train_test_split(X, y_name, test_size=0.2, random_state=42)
X_train, X_test, y_perfumes_train, y_perfumes_test = train_test_split(X, y_perfumes, test_size=0.2, random_state=42)
X_train, X_test, y_color_train, y_color_test = train_test_split(X, y_color, test_size=0.2, random_state=42)

# ----- เทรนโมเดล -----
rf_name = RandomForestClassifier(n_estimators=100, random_state=42)
rf_name.fit(X_train, y_name_train)

rf_perfumes = RandomForestClassifier(n_estimators=100, random_state=42)
rf_perfumes.fit(X_train, y_perfumes_train)

rf_color = RandomForestClassifier(n_estimators=100, random_state=42)
rf_color.fit(X_train, y_color_train)

# ----- บันทึกโมเดลและ LabelEncoders -----
flower_model_path = r"C:\Users\BIG GER\Desktop\IS\IS\model2"
joblib.dump(rf_name, f"{flower_model_path}\\rf_name.pkl")
joblib.dump(rf_perfumes, f"{flower_model_path}\\rf_perfumes.pkl")
joblib.dump(rf_color, f"{flower_model_path}\\rf_color.pkl")

# บันทึก LabelEncoder และ MultiLabelBinarizer
label_encoders = {
    'le_name': le_name,
    'le_perfumes': le_perfumes,
    'mlb': mlb
}
joblib.dump(label_encoders, f"{flower_model_path}\\label_encoders.pkl")
print("Saved label_encoders:", label_encoders)

# ----- รับ input จากผู้ใช้และทำนายค่าจาก input ใหม่ -----
def predict_flower():
    try:
        height = float(input("Enter flower height (cm): "))
        longevity = float(input("Enter flower longevity (years): "))
        
        input_data = pd.DataFrame([[height, longevity]], columns=X.columns)
        
        name_pred = le_name.inverse_transform(rf_name.predict(input_data))[0]
        perfumes_pred = le_perfumes.inverse_transform(rf_perfumes.predict(input_data))[0]
        color_pred = mlb.inverse_transform(rf_color.predict(input_data))
        
        result_message = f"\n===== Prediction Result =====\n🌸 Name: {name_pred}\n🌿 Has Perfume: {'Yes' if perfumes_pred else 'No'}\n🎨 Colors: {', '.join(color_pred[0]) if color_pred else 'Unknown'}"
        
        print(result_message)
        
    except ValueError:
        print("⚠ Invalid input! Please enter valid numeric values.")
        predict_flower()

# เรียกใช้ฟังก์ชัน
predict_flower()