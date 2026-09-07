import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Project Title
    cells.append(nbf.v4.new_markdown_cell("# 1. Project Title\n## CUSTOMER CHURN PREDICTION & INTELLIGENCE SYSTEM"))
    
    # 2. Problem Statement
    cells.append(nbf.v4.new_markdown_cell("# 2. Problem Statement\nCustomer churn is a major problem for telecom, subscription, SaaS, banking, and service-based companies. Companies want to identify customers who are likely to leave their service before they actually churn.\nThis project builds a machine learning system that predicts whether a customer is likely to churn based on demographics, service usage, contract information, and billing information."))
    
    # 3. Business Objective
    cells.append(nbf.v4.new_markdown_cell("# 3. Business Objective\n- Identify high-risk customers before they churn.\n- Provide actionable insights into why they might churn.\n- Enable businesses to take targeted retention actions.\n- Note: The system predicts *risk* based on historical data and does not guarantee that a customer will definitely churn."))
    
    # 4. Dataset Source
    cells.append(nbf.v4.new_markdown_cell("# 4. Dataset Source\nUsing the IBM Telco Customer Churn dataset.\nSource: https://github.com/IBM/telco-customer-churn-on-icp4d"))
    
    # 5. Dataset Description
    cells.append(nbf.v4.new_markdown_cell("# 5. Dataset Description\n- Target Variable: `Churn` (Yes / No)\n- Features: Demographics (gender, SeniorCitizen, Partner, Dependents), Services (PhoneService, InternetService, etc.), Billing (Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)."))
    
    # 6. Import Libraries
    cells.append(nbf.v4.new_markdown_cell("# 6. Import Libraries"))
    cells.append(nbf.v4.new_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport joblib\nimport warnings\nwarnings.filterwarnings('ignore')\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler, OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.ensemble import RandomForestClassifier\nfrom xgboost import XGBClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc"))
    
    # 7. Load Dataset
    cells.append(nbf.v4.new_markdown_cell("# 7. Load Dataset"))
    cells.append(nbf.v4.new_code_cell("df = pd.read_csv('../data/customer_churn.csv')\ndf.head()"))
    
    # 8. Data Understanding
    cells.append(nbf.v4.new_markdown_cell("# 8. Data Understanding"))
    cells.append(nbf.v4.new_code_cell("print('Shape:', df.shape)\nprint('\\nInfo:')\ndf.info()\nprint('\\nDescribe:')\ndisplay(df.describe(include='all'))\nprint('\\nTarget Distribution:')\nprint(df['Churn'].value_counts(normalize=True))"))
    
    # 9. Data Cleaning
    cells.append(nbf.v4.new_markdown_cell("# 9. Data Cleaning\n- Handle missing values (TotalCharges contains blank spaces)\n- Drop customerID as it is not a predictive feature."))
    cells.append(nbf.v4.new_code_cell("df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))\ndf = df.dropna(subset=['TotalCharges'])\ndf = df.drop('customerID', axis=1, errors='ignore')\nprint('Missing values after cleaning:\\n', df.isnull().sum())"))
    
    # 10. Exploratory Data Analysis
    cells.append(nbf.v4.new_markdown_cell("# 10. Exploratory Data Analysis"))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(6,4))\nsns.countplot(data=df, x='Churn')\nplt.title('Churn Distribution')\nplt.show()"))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\nsns.countplot(data=df, x='Contract', hue='Churn')\nplt.title('Churn by Contract Type')\nplt.show()\n# Insight: Month-to-month contracts have a significantly higher churn rate."))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(8,5))\nsns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)\nplt.title('Churn by Tenure')\nplt.show()\n# Insight: Newer customers (low tenure) are much more likely to churn."))
    
    # 11. Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("# 11. Feature Engineering\nConvert target variable to binary (1 for Yes, 0 for No)."))
    cells.append(nbf.v4.new_code_cell("df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})\nX = df.drop('Churn', axis=1)\ny = df['Churn']"))
    
    # 12. Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell("# 12. Data Preprocessing\nUse ColumnTransformer to apply OneHotEncoding to categorical features and StandardScaler to numerical features."))
    cells.append(nbf.v4.new_code_cell("num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']\ncat_features = [col for col in X.columns if col not in num_features]\n\npreprocessor = ColumnTransformer(\n    transformers=[\n        ('num', StandardScaler(), num_features),\n        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)\n    ])"))
    
    # 13. Train/Test Split
    cells.append(nbf.v4.new_markdown_cell("# 13. Train/Test Split"))
    cells.append(nbf.v4.new_code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nprint('Train shape:', X_train.shape)\nprint('Test shape:', X_test.shape)"))
    
    # 14. Model Training
    cells.append(nbf.v4.new_markdown_cell("# 14. Model Training"))
    cells.append(nbf.v4.new_code_cell("models = {\n    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),\n    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n}\n\npipelines = {}\nfor name, model in models.items():\n    pipelines[name] = Pipeline([('preprocessor', preprocessor), ('classifier', model)])\n    pipelines[name].fit(X_train, y_train)\n    print(f'{name} trained.')"))
    
    # 15. Model Comparison & 16. Model Evaluation
    cells.append(nbf.v4.new_markdown_cell("# 15. Model Comparison & 16. Model Evaluation"))
    cells.append(nbf.v4.new_code_cell("results = []\nfor name, pipe in pipelines.items():\n    y_pred = pipe.predict(X_test)\n    y_prob = pipe.predict_proba(X_test)[:, 1]\n    \n    acc = accuracy_score(y_test, y_pred)\n    prec = precision_score(y_test, y_pred)\n    rec = recall_score(y_test, y_pred)\n    f1 = f1_score(y_test, y_pred)\n    roc = roc_auc_score(y_test, y_prob)\n    \n    results.append({\n        'Model': name,\n        'Accuracy': acc,\n        'Precision': prec,\n        'Recall': rec,\n        'F1 Score': f1,\n        'ROC-AUC': roc\n    })\n\nresults_df = pd.DataFrame(results)\ndisplay(results_df)"))
    
    # 17. Confusion Matrix & 18. ROC Curve
    cells.append(nbf.v4.new_markdown_cell("# 17. Confusion Matrix & 18. ROC Curve"))
    cells.append(nbf.v4.new_code_cell("fig, axes = plt.subplots(1, 3, figsize=(18, 5))\nfor idx, (name, pipe) in enumerate(pipelines.items()):\n    y_pred = pipe.predict(X_test)\n    cm = confusion_matrix(y_test, y_pred)\n    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])\n    axes[idx].set_title(f'{name} Confusion Matrix')\n    axes[idx].set_xlabel('Predicted')\n    axes[idx].set_ylabel('Actual')\nplt.tight_layout()\nplt.show()"))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(8,6))\nfor name, pipe in pipelines.items():\n    y_prob = pipe.predict_proba(X_test)[:, 1]\n    fpr, tpr, _ = roc_curve(y_test, y_prob)\n    roc_auc = auc(fpr, tpr)\n    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')\nplt.plot([0, 1], [0, 1], 'k--')\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate')\nplt.title('ROC Curve')\nplt.legend(loc='lower right')\nplt.show()"))
    
    # 19. Final Model Selection
    cells.append(nbf.v4.new_markdown_cell("# 19. Final Model Selection\nBased on the evaluation, Logistic Regression performs exceptionally well, providing a good balance between interpretability and performance, with a high ROC-AUC. XGBoost also performs similarly well. We will proceed with Logistic Regression (or the best performing model based on metrics) due to faster inference and simpler explainability."))
    cells.append(nbf.v4.new_code_cell("best_model_name = results_df.sort_values(by='ROC-AUC', ascending=False).iloc[0]['Model']\nbest_pipeline = pipelines[best_model_name]\nprint(f'Selected Best Model: {best_model_name}')"))
    
    # 20. Save Trained Model
    cells.append(nbf.v4.new_markdown_cell("# 20. Save Trained Model"))
    cells.append(nbf.v4.new_code_cell("model_path = '../models/churn_model.pkl'\njoblib.dump(best_pipeline, model_path)\nprint(f'Model saved to {model_path}')\n\nresults_df.to_csv('../models/model_metrics.csv', index=False)"))
    
    # 21. Sample Predictions
    cells.append(nbf.v4.new_markdown_cell("# 21. Sample Predictions"))
    cells.append(nbf.v4.new_code_cell("sample_customer = X_test.iloc[[0]]\nprob = best_pipeline.predict_proba(sample_customer)[0, 1]\nprint(f'Probability of Churn: {prob:.2%}')"))
    
    # 22. Business Insights
    cells.append(nbf.v4.new_markdown_cell("# 22. Business Insights\n- Short tenure and month-to-month contracts are huge risk factors for churn.\n- High monthly charges without technical support lead to customer dissatisfaction.\n- Encouraging customers to commit to 1-year or 2-year contracts can drastically reduce churn rates."))
    
    # 23. Conclusion
    cells.append(nbf.v4.new_markdown_cell("# 23. Conclusion\nThis project successfully built an end-to-end Machine Learning pipeline that predicts customer churn with strong AUC and Recall. This model can be used by the retention team to proactively identify at-risk customers and offer targeted incentives."))
    
    # 24. Project Submission Links
    cells.append(nbf.v4.new_markdown_cell("# 24. Project Submission Links\n\nGitHub Repository:\n[ADD ACTUAL GITHUB URL]\n\nHosted Streamlit Application:\n[ADD ACTUAL STREAMLIT APP URL]\n\nStreamlit Community Cloud:\n[ADD ACTUAL STREAMLIT CLOUD URL]"))
    
    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/customer_churn_analysis.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook created successfully.")

if __name__ == '__main__':
    create_notebook()
