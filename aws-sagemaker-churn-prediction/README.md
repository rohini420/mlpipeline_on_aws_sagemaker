# Telco Customer Churn Prediction

A machine learning project that predicts customer churn for a telecommunications company using AWS SageMaker and XGBoost.

## Overview

This project uses the Telco Customer Churn dataset from Kaggle to build and deploy a binary classification model that predicts whether a customer will churn (leave the service) based on various customer attributes.

## Dataset

- **Source**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: 7,043 customers with 21 features
- **Target**: Churn (Yes/No)

## Features

The dataset includes customer demographics, account information, and service subscriptions:
- Customer demographics (gender, senior citizen status, partner, dependents)
- Account information (tenure, contract type, payment method, billing)
- Services (phone, internet, online security, tech support, streaming)
- Charges (monthly and total)

## Data Preprocessing

1. Removed `customerID` (non-predictive identifier)
2. Handled missing values in `TotalCharges` by filling with median
3. Converted target variable `Churn` to binary (0/1)
4. One-hot encoded all categorical variables
5. Converted boolean columns to integers for XGBoost compatibility
6. Split data: 80% training, 20% testing (stratified)

## Model Training

**Algorithm**: XGBoost (AWS SageMaker built-in)

**Hyperparameters**:
- Objective: `binary:logistic`
- Evaluation metric: `auc`
- Number of rounds: 200
- Max depth: 5
- Learning rate (eta): 0.2
- Subsample: 0.8
- Column sample by tree: 0.8

**Infrastructure**:
- Training instance: `ml.m5.large`
- Training duration: ~3 minutes

## Model Performance

Evaluated on the test set (1,409 samples):

| Metric | Score |
|--------|-------|
| **Accuracy** | 77.9% |
| **Precision** | 59.8% |
| **Recall** | 50.5% |
| **F1 Score** | 54.8% |
| **ROC AUC** | 81.5% |

**Confusion Matrix**:
```
[[908, 127],
 [185, 189]]
```

- True Negatives: 908 (correctly predicted non-churners)
- False Positives: 127 (predicted churn, but stayed)
- False Negatives: 185 (predicted stay, but churned)
- True Positives: 189 (correctly predicted churners)

## Deployment

The trained model is deployed as a real-time inference endpoint on AWS SageMaker:
- Endpoint instance: `ml.m5.large`
- Status: InService
- Input format: CSV (comma-separated feature values)
- Output format: JSON (prediction probabilities)

## Project Structure

```
├── sagemaker-exp.ipynb          # Main notebook with complete workflow
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── telco_train.csv              # Preprocessed training data
└── telco_test.csv               # Preprocessed test data
```

## Requirements

- Python 3.10+
- boto3
- sagemaker
- pandas
- scikit-learn
- kaggle (for dataset download)

## Usage

### 1. Setup Kaggle Credentials

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Install Dependencies

```bash
pip install kaggle boto3 sagemaker pandas scikit-learn
```

### 3. Run the Notebook

Execute all cells in `sagemaker-exp.ipynb` to:
- Download and preprocess data
- Train the XGBoost model
- Deploy the endpoint
- Make predictions

### 4. Make Predictions

```python
# Example prediction
sample_features = "0,72,114,8468,1,1,1,1,0,1,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,0"
prediction = predictor.predict(sample_features)

# Output format:
# {'predictions': [{'score': 0.0025536122266203165}]}
```

## Feature Order for Predictions

When making predictions, features must be provided in this order:
1. SeniorCitizen
2. tenure
3. MonthlyCharges
4. TotalCharges
5. gender_Male
6. Partner_Yes
7. Dependents_Yes
8. PhoneService_Yes
9. MultipleLines_No phone service
10. MultipleLines_Yes
11. InternetService_Fiber optic
12. InternetService_No
13. OnlineSecurity_No internet service
14. OnlineSecurity_Yes
15. OnlineBackup_No internet service
16. OnlineBackup_Yes
17. DeviceProtection_No internet service
18. DeviceProtection_Yes
19. TechSupport_No internet service
20. TechSupport_Yes
21. StreamingTV_No internet service
22. StreamingTV_Yes
23. StreamingMovies_No internet service
24. StreamingMovies_Yes
25. Contract_One year
26. Contract_Two year
27. PaperlessBilling_Yes
28. PaymentMethod_Credit card (automatic)
29. PaymentMethod_Electronic check
30. PaymentMethod_Mailed check

## Storage

- **S3 Bucket**: Training and test data stored in SageMaker's default bucket
- **Model Artifacts**: Stored in S3 under the training job's output path
- **Path Structure**: `s3://<bucket>/telco-churn/train/` and `s3://<bucket>/telco-churn/test/`

## Model Insights

- The model achieves a good **ROC AUC of 81.5%**, indicating strong discriminative ability
- **Recall of 50.5%** suggests the model is conservative, missing about half of actual churners
- **Precision of 59.8%** means when it predicts churn, it's correct about 60% of the time
- Consider adjusting the prediction threshold (default 0.5) based on business costs:
  - Lower threshold (e.g., 0.3) to catch more churners (higher recall, lower precision)
  - Higher threshold (e.g., 0.7) for more confident predictions (lower recall, higher precision)

## Next Steps

Potential improvements:
1. **Feature Engineering**: Create interaction features, tenure buckets, or aggregate service counts
2. **Class Imbalance**: Try SMOTE, class weights, or different sampling strategies
3. **Hyperparameter Tuning**: Use SageMaker's automatic model tuning
4. **Alternative Models**: Test other algorithms like Random Forest or Neural Networks
5. **Cost-Sensitive Learning**: Incorporate business costs of false positives vs false negatives

## Cleanup

To avoid ongoing charges, remember to:
```python
# Delete the endpoint
predictor.delete_endpoint()

# Or via AWS CLI
aws sagemaker delete-endpoint --endpoint-name <endpoint-name>
```

## License

Dataset license: Copyright Authors (as specified on Kaggle)

---

**Note**: This is an educational project. Ensure you have proper AWS permissions and monitor your resource usage to avoid unexpected charges.