import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def retrain_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "customer_churn.csv")
    df = pd.read_csv(data_path)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
    df = df.dropna(subset=['TotalCharges'])
    df = df.drop('customerID', axis=1, errors='ignore')
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = [col for col in X.columns if col not in num_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)
        ])
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        pipelines[name].fit(X_train, y_train)
        print(f'{name} trained.')
        
    results = []
    for name, pipe in pipelines.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': roc
        })

    results_df = pd.DataFrame(results)
    
    best_model_name = results_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]['Model']
    best_pipeline = pipelines[best_model_name]
    print(f'Selected Best Model: {best_model_name}')
    
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'churn_model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f'Model saved to {model_path}')
    
    results_df.to_csv(os.path.join(models_dir, 'model_metrics.csv'), index=False)
    print('Metrics saved.')

if __name__ == '__main__':
    retrain_model()
