import joblib
import pandas as pd

# Load the saved model
model = joblib.load(r"C:\Users\sukha\OneDrive\Desktop\parkinsons-ml\models\parkinsons_best_model.pkl")

# Example input (replace these with real patient data)
input_data = {
    'MDVP:Fo(Hz)': [120.0],
    'MDVP:Fhi(Hz)': [140.0],
    'MDVP:Flo(Hz)': [100.0],
    'MDVP:Jitter(%)': [0.005],
    'MDVP:Jitter(Abs)': [0.00003],
    'MDVP:RAP': [0.0025],
    'MDVP:PPQ': [0.003],
    'Jitter:DDP': [0.007],
    'MDVP:Shimmer': [0.03],
    'MDVP:Shimmer(dB)': [0.25],
    'Shimmer:APQ3': [0.015],
    'Shimmer:APQ5': [0.02],
    'MDVP:APQ': [0.02],
    'Shimmer:DDA': [0.045],
    'NHR': [0.02],
    'HNR': [20.0],
    'RPDE': [0.45],
    'DFA': [0.6],
    'spread1': [-5.0],
    'spread2': [0.35],
    'D2': [2.0],
    'PPE': [0.3]
}

# Convert to DataFrame
sample_df = pd.DataFrame(input_data)

# Predict
prediction = model.predict(sample_df)[0]

if prediction == 1:
    print("🧠 The model predicts: Person **HAS Parkinson’s Disease**.")
else:
    print("✅ The model predicts: Person **DOES NOT have Parkinson’s Disease**.")
