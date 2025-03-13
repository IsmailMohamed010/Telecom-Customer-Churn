# Telco Customer Churn Prediction

This project aims to build a machine-learning model to predict customer churn for a telecom company. By analyzing customer data, service subscriptions, and payment details, we can identify factors contributing to churn and provide actionable insights for retention strategies.

## Dataset
We used the **Telco Customer Churn Dataset** from Kaggle, which contains customer information, service details, contract types, payment details, and churn status.

### Dataset Download
[Click here to download the dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Project Structure
```
.
├── data/               # Dataset files
├── notebooks/           # Jupyter Notebooks for analysis and modeling
├── models/              # Saved machine learning models
├── src/                 # Python scripts for preprocessing and evaluation
└── README.md            # Project documentation (this file)
```

## Usage
1. **Download the dataset:** Place the downloaded CSV file in the `data/` directory.
2. **Run Jupyter Notebooks:** Launch the notebooks for preprocessing, feature engineering, and model training:
   ```bash
   jupyter notebook
   ```

## Key Steps
1. **Data Preprocessing:**
   - Handle missing values in the `TotalCharges` column.
   - Encode categorical variables using One-Hot Encoding.
   - Scale numerical features for optimal model performance.

2. **Exploratory Data Analysis (EDA):**
   - Analyze factors affecting churn using data visualization.
   - Correlation analysis and feature importance.

3. **Model Building:**
   - Train models including Logistic Regression, Decision Trees, and Gradient Boosting.
   - Perform hyperparameter tuning for better accuracy.

4. **Model Evaluation:**
   - Evaluate models using classification metrics such as Accuracy, Precision, Recall, and F1-Score.
   - Visualize confusion matrix and classification report.

## Results
- Achieved an accuracy of **80%** with Gradient Boosting.
- Identified critical factors contributing to churn, such as contract type and payment methods.

## Future Improvements
- Implement advanced algorithms such as XGBoost for enhanced performance.
- Address class imbalance with techniques like SMOTE.
- Explore feature interactions and customer segmentation.

## Example Code
Here is a sample code snippet to visualize the confusion matrix:
```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Assume y_test and y_pred are defined
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()
```

## Contributing
We welcome contributions to this project. Feel free to submit issues or create pull requests.
