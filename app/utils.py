def get_model_accuracy():
    """Return model accuracy (actual or manually set)."""
    return 89.74  # Random Forest accuracy from your results

def get_resource_links():
    """Useful links for sidebar."""
    return {
        "Parkinson’s Foundation": "https://www.parkinson.org/",
        "Oxford Parkinson's Dataset (UCI/Kaggle)": "https://www.kaggle.com/datasets/thecansin/parkinsons-data-set",
        "WHO: Parkinson’s Disease Info": "https://www.who.int/news-room/fact-sheets/detail/parkinson-disease"
    }

def get_confidence_score(proba, prediction):
    """Return confidence percentage for predicted class."""
    return round(float(proba[prediction]) * 100, 2)
