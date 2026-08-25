import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">❤️ Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Prediction System</div>',
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    url = (
        "https://archive.ics.uci.edu/"
        "ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target"
    ]

    df = pd.read_csv(
        url,
        names=columns,
        na_values="?"
    )

    return df


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model():

    df = load_data()

    # Convert columns to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Handle missing values
    imputer = SimpleImputer(
        strategy="median"
    )

    X = imputer.fit_transform(X)

    # Convert target to binary
    # 0 = No heart disease
    # 1 = Heart disease

    y = y.apply(
        lambda value: 1 if value > 0 else 0
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return model, scaler, accuracy


# =========================================================
# TRAIN MODEL
# =========================================================

try:

    model, scaler, accuracy = train_model()

except Exception as e:

    st.error(
        "Unable to load the dataset. "
        "Please check your internet connection."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Model Information")

st.sidebar.write(
    "Machine Learning Algorithm:"
)

st.sidebar.success(
    "Random Forest Classifier"
)

st.sidebar.write(
    "Model Accuracy:"
)

st.sidebar.success(
    f"{accuracy * 100:.2f}%"
)

st.sidebar.write(
    "Dataset:"
)

st.sidebar.info(
    "UCI Heart Disease Dataset"
)


# =========================================================
# PATIENT INPUT
# =========================================================

st.header("👤 Enter Patient Information")

col1, col2 = st.columns(2)


# =========================================================
# COLUMN 1
# =========================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        [
            "Female",
            "Male"
        ]
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol (mg/dl)",
        min_value=50,
        max_value=700,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [
            "No",
            "Yes"
        ]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )


# =========================================================
# COLUMN 2
# =========================================================

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [
            "No",
            "Yes"
        ]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [
            0,
            1,
            2,
            3
        ]
    )

    thal = st.selectbox(
        "Thalassemia",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )


# =========================================================
# CONVERT INPUT VALUES
# =========================================================

sex_value = 1 if sex == "Male" else 0


cp_value = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-anginal Pain": 3,
    "Asymptomatic": 4
}[cp]


fbs_value = 1 if fbs == "Yes" else 0


restecg_value = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}[restecg]


exang_value = 1 if exang == "Yes" else 0


slope_value = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}[slope]


thal_value = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}[thal]


# =========================================================
# CREATE INPUT DATA
# =========================================================

input_data = np.array([[
    age,
    sex_value,
    cp_value,
    trestbps,
    chol,
    fbs_value,
    restecg_value,
    thalach,
    exang_value,
    oldpeak,
    slope_value,
    ca,
    thal_value
]])


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Scale input
    input_scaled = scaler.transform(
        input_data
    )

    # Prediction
    prediction = model.predict(
        input_scaled
    )[0]

    # Probability
    probability = model.predict_proba(
        input_scaled
    )[0][1]

    st.divider()

    st.header("📋 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Higher likelihood of heart disease"
        )

        st.write(
            f"Model-estimated probability: "
            f"**{probability * 100:.2f}%**"
        )

    else:

        st.success(
            "✅ Lower likelihood of heart disease"
        )

        st.write(
            f"Model-estimated probability: "
            f"**{probability * 100:.2f}%**"
        )

    # Probability bar
    st.progress(
        float(probability)
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.divider()

st.warning(
    "⚠️ Disclaimer: This application is an educational "
    "machine-learning project. It is not a medical "
    "diagnostic tool. Please consult a qualified healthcare "
    "professional for medical advice."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    "<center>Heart Disease Prediction using Machine Learning</center>",
    unsafe_allow_html=True
)
