import streamlit as st
import joblib
import numpy as np
import pandas as pd
from utils import get_model_accuracy, get_resource_links, get_confidence_score

# -------------------------------
# 🎯 App Configuration
# -------------------------------
st.set_page_config(
    page_title="Parkinson's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
    
)

# -------------------------------
# 🎨 Load Custom CSS
# -------------------------------
try:
    with open("app/style.css", "r", encoding="utf-8") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ style.css not found — using default Streamlit style.")

# -------------------------------
# 🧠 Load Model
# -------------------------------
model = joblib.load("models/parkinsons_best_model.pkl")

# -------------------------------
# 🏷️ Sidebar
# -------------------------------
st.sidebar.title("🧩 Model Information")
st.sidebar.markdown(f"**Accuracy:** {get_model_accuracy()}%")
st.sidebar.markdown("**Algorithm Used:** Logistic Regression / Random Forest Hybrid")
st.sidebar.markdown("**Dataset:** Oxford Parkinson's Disease Detection Dataset")
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Resources")
for name, link in get_resource_links().items():
    st.sidebar.markdown(f"[{name}]({link})")
st.sidebar.markdown("---")
st.sidebar.caption("Developed with ❤️ using Streamlit and Scikit-Learn")

# -------------------------------
# 🏷️ Main Title
# -------------------------------
st.title("🧠 Parkinson's Disease Prediction App")
st.write("""
Enter your voice measurement features below to estimate the likelihood of Parkinson’s Disease.  
The model will predict whether the pattern is *Healthy* or *Parkinsonian* and show a confidence score.
""")

# -------------------------------
# 🌙 Dark Mode Toggle
# -------------------------------
dark_mode = st.toggle("🌙 Enable Dark Mode")

if dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #121212; color: #e0e0e0; }
        [data-testid="stSidebar"] { background-color: #1e1e1e !important; color: #fff; }
        .stButton>button { background: linear-gradient(90deg, #444, #888); color: white; }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# 📄 Load Healthy Mean Values
# -------------------------------
try:
    healthy_mean = pd.read_csv("data/healthy_mean.csv", index_col=0).squeeze("columns").to_dict()
    st.sidebar.success("✅ Loaded Healthy Mean defaults from CSV")
except Exception as e:
    st.sidebar.error(f"⚠️ Could not load healthy_mean.csv — using fallback values.")
    healthy_mean = {
        "MDVP:Fo(Hz)": 181.937771, "MDVP:Fhi(Hz)": 223.636750, "MDVP:Flo(Hz)": 145.207292,
        "MDVP:Jitter(%)": 0.003866, "MDVP:Jitter(Abs)": 0.000023, "MDVP:RAP": 0.001925,
        "MDVP:PPQ": 0.002056, "Jitter:DDP": 0.005776, "MDVP:Shimmer": 0.017615,
        "MDVP:Shimmer(dB)": 0.162958, "Shimmer:APQ3": 0.009504, "Shimmer:APQ5": 0.010509,
        "MDVP:APQ": 0.013305, "Shimmer:DDA": 0.028511, "NHR": 0.011483, "HNR": 24.678750,
        "RPDE": 0.442552, "DFA": 0.695716, "spread1": -6.759264, "spread2": 0.160292,
        "D2": 2.154491, "PPE": 0.123017
    }

st.sidebar.info("Using Healthy Mean values as default for all input fields.")

# -------------------------------
# 🧩 Input Features
# -------------------------------
features = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE"
]

feature_ranges = {
    "MDVP:Fo(Hz)": (80.0, 300.0), "MDVP:Fhi(Hz)": (100.0, 400.0), "MDVP:Flo(Hz)": (60.0, 250.0),
    "MDVP:Jitter(%)": (0.0, 0.02), "MDVP:Jitter(Abs)": (0.0, 0.001), "MDVP:RAP": (0.0, 0.02),
    "MDVP:PPQ": (0.0, 0.02), "Jitter:DDP": (0.0, 0.05), "MDVP:Shimmer": (0.0, 0.1),
    "MDVP:Shimmer(dB)": (0.0, 1.0), "Shimmer:APQ3": (0.0, 0.05), "Shimmer:APQ5": (0.0, 0.05),
    "MDVP:APQ": (0.0, 0.05), "Shimmer:DDA": (0.0, 0.1), "NHR": (0.0, 0.3),
    "HNR": (0.0, 40.0), "RPDE": (0.0, 1.0), "DFA": (0.5, 1.0),
    "spread1": (-8.0, -2.0), "spread2": (0.0, 0.5), "D2": (1.0, 4.0), "PPE": (0.0, 1.0)
}

col1, col2 = st.columns(2)
inputs = []

for i, feature in enumerate(features):
    with col1 if i % 2 == 0 else col2:
        min_val, max_val = feature_ranges[feature]
        default_val = healthy_mean.get(feature, np.mean([min_val, max_val]))
        value = st.number_input(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=float(default_val),
            format="%.6f"
        )
        inputs.append(value)

# -------------------------------
# 🔮 Prediction
# -------------------------------
if st.button("🔍 Predict Parkinson’s Status", use_container_width=True):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    proba = model.predict_proba(input_array)[0]

    confidence = get_confidence_score(proba, prediction)

    if prediction == 1:
        st.error(f"🧠 Parkinson’s Detected — Model Confidence: **{confidence}%**")
        st.progress(proba[1])
    else:
        st.success(f"✅ Healthy — Model Confidence: **{confidence}%**")
        st.progress(proba[0])

    # 📊 Probability Chart
    st.markdown("### 📊 Prediction Probabilities")
    chart_data = pd.DataFrame({
        "Status": ["Healthy", "Parkinson’s"],
        "Probability": proba
    })
    st.bar_chart(chart_data.set_index("Status"))
