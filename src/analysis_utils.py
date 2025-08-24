# src/analysis_utils.py
"""
Advanced analysis functions: machine learning and prediction models.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def predict_fine_payment(df):
    """
    Example ML model: Predict if a fine will be paid based on features.
    Args:
        df (DataFrame): Cleaned dataset.
    Returns:
        str: Classification report.
    """
    # Drop missing target
    df = df.dropna(subset=['Fine_Paid'])

    # Feature and target
    X = df[['Driver_Age']]   # simple example
    y = df['Fine_Paid'].apply(lambda x: 1 if x=="Yes" else 0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    return classification_report(y_test, y_pred)
