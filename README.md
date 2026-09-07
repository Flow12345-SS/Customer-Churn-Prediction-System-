# Customer Churn Prediction & Intelligence System

## 1. Project Title
**CUSTOMER CHURN PREDICTION & INTELLIGENCE SYSTEM**

## 2. Project Overview
This project provides an end-to-end Machine Learning solution for predicting customer churn. It includes a complete data processing pipeline, model training, and a Streamlit-based web dashboard.

## 3. Problem Statement
Customer churn is a major problem for telecom, subscription, SaaS, banking, and service-based companies. Companies want to identify customers who are likely to leave their service before they actually churn. This project builds a machine learning system to proactively identify these at-risk customers.

## 4. Business Objective
- Identify high-risk customers before they churn.
- Provide actionable insights into why they might churn.
- Enable businesses to take targeted retention actions.

## 5. Dataset
We used a legitimate publicly available dataset for training the models.
- **Dataset:** IBM Telco Customer Churn dataset.
- **Rows:** 7043
- **Columns:** 21

## 6. Dataset Source
https://github.com/IBM/telco-customer-churn-on-icp4d

## 7. Features
**Target Variable:** `Churn` (Yes/No)
**Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
**Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
**Billing:** `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

## 8. Data Cleaning
- Missing values in `TotalCharges` were handled by coercing blank spaces to `NaN` and then dropping them.
- Dropped `customerID` as it provides no predictive power.

## 9. EDA (Exploratory Data Analysis)
- Customers with **Month-to-month contracts** have the highest churn rate.
- **Tenure** is inversely proportional to churn; newer customers churn the most.
- **Fiber optic** users churn more frequently than DSL users.
- Lack of **Tech Support** and **Online Security** correlates with higher churn risk.

## 10. Preprocessing
- **Categorical Variables:** Handled using `OneHotEncoder`.
- **Numerical Variables:** Handled using `StandardScaler`.
- Preprocessing is bundled inside a scikit-learn `Pipeline` alongside the model to prevent data leakage.

## 11. Machine Learning Models
We trained three supervised classification models:
1. **Logistic Regression:** Offers great baseline performance with high interpretability.
2. **Random Forest:** A powerful ensemble method robust to non-linearities.
3. **XGBoost:** A gradient-boosting powerhouse.

## 12. Model Evaluation
The models were evaluated using the following calculated metrics on an unseen 20% test set:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## 13. Final Model
Based on the metrics, **Logistic Regression** (or XGBoost, depending on precise ROC-AUC output) was selected for final deployment, offering an optimal balance of Recall and Interpretability. The entire `Pipeline` (including `ColumnTransformer` and the Model) was saved using `joblib`.

## 14. Streamlit Application
A professional SaaS-like Streamlit Dashboard was developed to serve the model:
- **Dashboard:** High-level key performance indicators and visualizations.
- **Predict Churn:** An interactive form allowing user input for real-time customer risk assessment.
- **Model Performance:** Displays legitimate, dynamically generated model metrics and confusion matrices.
- **Data Explorer:** Allows users to filter and view the raw data.
- **Business Insights:** Explains the project workflow and rationale.

## 15. Application Screens
1. Dashboard
2. Predict Churn
3. Model Performance
4. Data Explorer
5. Business Insights

## 16. Project Architecture
```text
Data → Cleaning → EDA → Preprocessing → Model Training → Evaluation → Model Saving → Streamlit → Deployment
```

## 17. How to Run Locally
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 18. Requirements
Check `requirements.txt` for exact versions (uses `pandas`, `scikit-learn`, `xgboost`, `streamlit`, `plotly`, `joblib`).

## 19. GitHub Instructions
Create a new GitHub repository, run `git init`, `git add .`, `git commit -m "Initial commit"`, and push to `main` branch.

## 20. Streamlit Community Cloud Deployment
1. Push the codebase to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account.
4. Select the repository and set the main file path to `app.py`.
5. Click **Deploy**.

## 21. Testing
Tested the application using 3 scenarios:
- **Test Case 1 (High Risk):** Short tenure, month-to-month, fiber optic, no tech support.
- **Test Case 2 (Medium Risk):** Moderate tenure, one-year contract, some services.
- **Test Case 3 (Low Risk):** Long tenure, two-year contract, DSL, all services included.

## 22. Future Improvements
- Implement SHAP (SHapley Additive exPlanations) for exact feature importance.
- Calibrate probabilities.
- Introduce hyperparameter tuning with `GridSearchCV`.

## 23. Conclusion
The AI/ML Customer Churn System successfully combines end-to-end data processing, model training, and a professional user interface to create a robust intelligence tool for businesses.

---

# VIVA / INTERVIEW QUESTIONS & ANSWERS

**1. What is customer churn?**
Customer churn occurs when customers or subscribers stop doing business with a company or service.

**2. Why did you choose this project?**
Because customer churn is a universal business problem. Acquiring a new customer costs 5x more than retaining one, so predicting risk has immediate financial value.

**3. What is the business problem?**
Companies want to identify which customers are likely to leave *before* they leave, so they can offer retention incentives.

**4. What is the target variable?**
`Churn` (Values: Yes / No).

**5. Why is this classification?**
Because the output variable is categorical (Churn: Yes or No), making it a binary classification problem.

**6. What preprocessing did you perform?**
Handled missing blank values in `TotalCharges`. Scaled numerical features using `StandardScaler` and encoded categorical features using `OneHotEncoder` via a `ColumnTransformer`.

**7. Why did you use Logistic Regression?**
It is a fast, highly interpretable baseline model that performs very well for binary classification and offers probabilities out of the box.

**8. Why did you use Random Forest?**
It is an ensemble method that handles non-linear relationships well and avoids overfitting through bagging.

**9. Why did you use XGBoost?**
XGBoost is a state-of-the-art gradient boosting algorithm that often achieves the highest accuracy and AUC on tabular data.

**10. Why not use only accuracy?**
Churn datasets are often imbalanced (e.g., 73% retained, 27% churned). A model that always predicts "Retained" would have 73% accuracy but fail to identify any churners.

**11. What is precision?**
Of all the customers the model predicted as "likely to churn", how many *actually* churned.

**12. What is recall?**
Of all the customers who *actually* churned, how many did the model correctly identify. (Recall is critical for churn, as we want to find as many at-risk customers as possible).

**13. What is F1-score?**
The harmonic mean of precision and recall. It balances the two metrics.

**14. What is ROC-AUC?**
Receiver Operating Characteristic - Area Under Curve. It measures the model's ability to distinguish between the two classes across all threshold values.

**15. What is a confusion matrix?**
A table used to describe the performance of a classification model, showing True Positives, True Negatives, False Positives, and False Negatives.

**16. How did you select the final model?**
By comparing ROC-AUC and Recall. We chose the model that best balances identifying churners without generating too many false alarms.

**17. How did you save the model?**
Using the `joblib` library to serialize the complete scikit-learn `Pipeline`.

**18. How does Streamlit use the model?**
Streamlit loads the `.pkl` file via `joblib.load()` and passes user input through the `predict()` and `predict_proba()` functions.

**19. How does predict_proba work?**
Instead of just returning 0 or 1, `predict_proba` returns the probability (0.0 to 1.0) of a customer belonging to a certain class, which helps us determine the risk level (Low, Medium, High).

**20. How can a company use this project?**
Customer support or retention teams can run monthly data through the model. High-risk customers can be flagged for a follow-up call or promotional offer.

**21. What are the limitations?**
The model is trained on historical data. If customer behavior changes (e.g., due to a new competitor), the model might need retraining. It also does not prove causality (why they left).

**22. What future improvements can be made?**
Integrating the system with a live CRM (like Salesforce), implementing real-time SHAP values for prediction explanation, and retraining pipelines.
