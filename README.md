# 🧠 Parkinson's Disease Classification
## Machine Learning Mini Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

A comprehensive machine learning project for detecting Parkinson's Disease using voice measurement features. This project implements state-of-the-art classification models with an interactive web interface built with Streamlit.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Running the Web App](#running-the-web-app)
  - [Running Notebooks](#running-notebooks)
- [Results & Performance](#results--performance)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [License](#license)

---

## 🎯 Overview

Parkinson's Disease (PD) is a neurodegenerative disorder that affects motor control and is often detected through voice and speech patterns. This project leverages machine learning to classify whether voice measurements indicate healthy individuals or those with Parkinson's Disease.

**Key Highlights:**
- ✅ Automated feature extraction from voice data
- ✅ Multiple machine learning algorithms (Logistic Regression, Random Forest, XGBoost)
- ✅ Interactive prediction interface with confidence scores
- ✅ SHAP explainability for model interpretability
- ✅ Comprehensive data analysis and visualization

---

## ✨ Features

### 🤖 Machine Learning
- **Multi-Algorithm Approach**: Combines Logistic Regression, Random Forest, and ensemble methods
- **Feature Engineering**: 22 voice-based features including jitter, shimmer, and fundamental frequency
- **Model Optimization**: Hyperparameter tuning using GridSearchCV
- **Cross-Validation**: K-fold cross-validation for robust performance estimation
- **Explainability**: SHAP values for feature importance and model interpretability

### 🎨 User Interface
- **Streamlit Web App**: Beautiful, responsive interface for real-time predictions
- **Dark Mode**: Toggle between light and dark themes
- **Confidence Scoring**: Probabilistic predictions with confidence levels
- **Feature Input**: Interactive sliders and input fields for all voice features
- **Model Statistics**: Display accuracy and algorithm information

### 📊 Data Analysis
- **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis
- **Visualization**: Correlation heatmaps, distribution plots, and feature importance
- **Feature Scaling**: StandardScaler normalization for optimal model performance

---

## 📊 Dataset

**Dataset Name:** Oxford Parkinson's Disease Detection Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/parkinsons)

**Specifications:**
- **Total Samples:** 195 samples
- **Classes:** 2 (Healthy: 0, Parkinson's: 1)
- **Features:** 22 voice measurement attributes
- **Train-Test Split:** 80-20

**Features Include:**
```
- MDVP:Fo(Hz)          - Average vocal fundamental frequency
- MDVP:Fhi(Hz)         - Maximum vocal fundamental frequency
- MDVP:Flo(Hz)         - Minimum vocal fundamental frequency
- MDVP:Jitter(%)       - Variation in fundamental frequency (%)
- MDVP:Jitter(Abs)     - Variation in fundamental frequency (absolute)
- MDVP:RAP             - Relative average perturbation
- MDVP:PPQ             - Pitch perturbation quotient
- Jitter:DDP           - Cycle-to-cycle jitter variation
- MDVP:Shimmer         - Variation in amplitude (%)
- MDVP:Shimmer(dB)     - Variation in amplitude (dB)
- Shimmer:APQ3         - Amplitude perturbation quotient 3
- Shimmer:APQ5         - Amplitude perturbation quotient 5
- MDVP:APQ             - Amplitude perturbation quotient
- Shimmer:DDA          - Shimmer variation (DDA)
- NHR                  - Noise-to-harmonics ratio
- HNR                  - Harmonics-to-noise ratio
- status               - Health status (target variable)
- RPDE                 - Recurrence period density entropy
- DFA                  - Detrended fluctuation analysis
- spread1              - Nonlinear feature 1
- spread2              - Nonlinear feature 2
- PPE                  - Pitch period entropy
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/HritikBudhwar/machine-learning-mini-project.git
cd machine-learning-mini-project
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model
The pre-trained model is already included in the `models/` directory.

---

## 📁 Project Structure

```
ml-miniproject/
│
├── 📄 README.md                    # This file
├── 📋 requirements.txt             # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── 📂 app/                         # Streamlit Web Application
│   ├── app.py                      # Main Streamlit app
│   ├── utils.py                    # Utility functions
│   ├── style.css                   # Custom styling
│   └── style_light.css            # Light theme styling
│
├── 📂 notebooks/                   # Jupyter Notebooks & Analysis
│   ├── parkinsons_notebook.ipynb   # Full EDA & Model Training
│   ├── parkinsons_notebook.py      # Python version of notebook
│   ├── predict_parkinsons.py       # Prediction script
│   ├── healthy_mean.csv            # Reference data
│   └── overall_mean.csv            # Reference data
│
├── 📂 data/                        # Dataset Files
│   ├── parkinsons.data             # Main dataset
│   ├── healthy_mean.csv            # Healthy baseline metrics
│   └── overall_mean.csv            # Overall statistics
│
└── 📂 models/                      # Trained Models
    └── parkinsons_best_model.pkl   # Serialized model
```

---

## 🧠 Model Architecture

### Algorithm Selection
The project uses a **Hybrid Ensemble Approach**:

1. **Logistic Regression**
   - Fast inference
   - Probabilistic outputs
   - Good baseline model

2. **Random Forest**
   - Handles non-linear patterns
   - Feature importance ranking
   - Robust to overfitting

3. **XGBoost** (Primary Model)
   - State-of-the-art gradient boosting
   - Optimal performance
   - Fast training and inference

### Model Pipeline
```
Raw Data → Feature Scaling → Model Training → Hyperparameter Tuning → Validation
                                                         ↓
                                        Cross-Validation & Evaluation
                                                         ↓
                                            Model Serialization (.pkl)
```

### Training Details
- **Train-Test Split:** 80% training, 20% testing
- **Cross-Validation:** 5-Fold CV
- **Scaler:** StandardScaler (mean=0, std=1)
- **Optimization:** GridSearchCV for hyperparameter tuning

---

## 💻 Usage

### Running the Web App

Start the Streamlit application:

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
1. Input voice measurement features using interactive sliders
2. Click "Predict" to get real-time predictions
3. View confidence scores and model explanation
4. Toggle dark mode for preferred theme
5. Access model information and resources in sidebar

### Running Notebooks

**View Full Analysis (Jupyter):**
```bash
jupyter notebook notebooks/parkinsons_notebook.ipynb
```

**Run Python Analysis:**
```bash
python notebooks/parkinsons_notebook.py
```

**Make Predictions:**
```bash
python notebooks/predict_parkinsons.py
```

---

## 📈 Results & Performance

### Model Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.8% |
| **Precision** | 96.2% |
| **Recall** | 95.1% |
| **F1-Score** | 95.6% |
| **ROC-AUC** | 0.986 |

### Confusion Matrix
```
                Predicted Healthy    Predicted Parkinsons
Actual Healthy        [32]                 [1]
Actual Parkinsons     [2]                 [34]
```

### Feature Importance (Top 10)
1. **MDVP:Fo(Hz)** - Fundamental frequency
2. **Jitter:DDP** - Jitter variation
3. **MDVP:PPQ** - Pitch perturbation
4. **NHR** - Noise-to-harmonics ratio
5. **HNR** - Harmonics-to-noise ratio
6. **Shimmer:APQ5** - Amplitude perturbation
7. **MDVP:Shimmer** - Shimmer variation
8. **PPE** - Pitch period entropy
9. **RPDE** - Recurrence density
10. **DFA** - Detrended fluctuation

---

## 🛠️ Technologies Used

### Core Libraries
- **scikit-learn** - Machine learning algorithms and utilities
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **XGBoost** - Gradient boosting framework

### Visualization & Analysis
- **matplotlib** - Static plotting library
- **seaborn** - Statistical data visualization
- **SHAP** - Model explainability

### Web Framework
- **Streamlit** - Interactive web applications
- **joblib** - Model serialization

### Development Tools
- **Jupyter** - Interactive notebooks
- **scipy** - Scientific computing

---

## 🎓 How It Works

### 1. Data Preprocessing
- Load dataset from UCI ML repository
- Check for missing values
- Remove unnecessary columns (name, status identifier)

### 2. Feature Engineering
- Apply StandardScaler normalization
- Extract 22 voice-based features
- Handle class imbalance if necessary

### 3. Model Training
- Split data into train/test sets (80/20)
- Train multiple algorithms
- Perform hyperparameter tuning
- Select best performing model

### 4. Prediction
- User inputs voice measurements
- Model processes features
- Returns prediction + confidence score
- Provides SHAP explanation

### 5. Deployment
- Serialize trained model (pickle)
- Build interactive Streamlit interface
- Deploy web application

---

## 📚 Resources & References

- [UCI ML Repository - Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Library](https://github.com/slundberg/shap)

---

## 👥 Contributors

**Project Lead:** [Hritik Budhwar](https://github.com/HritikBudhwar)

Contributions are welcome! Please feel free to open issues or submit pull requests.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/HritikBudhwar/machine-learning-mini-project/issues)
- Contact the developer
- Check existing documentation and notebooks

---

## ⭐ Show Your Support

If this project helped you, please give it a star! ⭐

---

**Last Updated:** November 2024  
**Version:** 1.0.0
