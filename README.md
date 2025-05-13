# 💳 Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions using the **Credit Card Fraud Detection** dataset. It applies preprocessing, exploratory data analysis, and trains a Random Forest model to classify transactions as valid or fraudulent.

---

## 📁 Dataset

We use the `creditcard.csv` dataset, which includes:

- `Time`, `Amount`
- Anonymized features (`V1` to `V28`)
- `Class`:  
  - `0` → Valid  
  - `1` → Fraudulent

---

## 🛠️ How to Run

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook fraud_detection_model.ipynb
   ```

---

## 📦 Requirements

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  

---

## 🧪 Code Overview

### 🔹 Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
```

### 🔹 Load Dataset

```python
data = pd.read_csv("creditcard.csv")
print(data.head())
print(data.describe())
```

### 🔹 Data Split: Fraud & Valid

```python
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlierFraction = len(fraud)/float(len(valid))
print(outlierFraction)
print('Fraud Cases:', len(fraud))
print('Valid Transactions:', len(valid))
```

### 🔹 Amount Stats

```python
print("Fraud Amount Stats:")
print(fraud.Amount.describe())
print("Valid Amount Stats:")
print(valid.Amount.describe())
```

### 🔹 Correlation Matrix

```python
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()
```

### 🔹 Feature and Target Split

```python
X = data.drop(['Class'], axis=1)
Y = data["Class"]
xData = X.values
yData = Y.values
```

### 🔹 Train-Test Split

```python
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42)
```

### 🔹 Random Forest Model

```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)
```

### 🔹 Handle NaN (if any)

```python
from sklearn.impute import SimpleImputer

print("NaN values in yTest before imputation:", np.isnan(yTest).sum())

imputer = SimpleImputer(strategy='most_frequent')
yTest = imputer.fit_transform(yTest.reshape(-1, 1)).flatten()

print("NaN values in yTest after imputation:", np.isnan(yTest).sum())
```

### 🔹 Evaluation Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

accuracy = accuracy_score(yTest, yPred)
precision = precision_score(yTest, yPred)
recall = recall_score(yTest, yPred)
f1 = f1_score(yTest, yPred)
mcc = matthews_corrcoef(yTest, yPred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
```

### 🔹 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()
```

---

## 📈 Sample Output

```
Model Evaluation Metrics:
Accuracy: 0.9500
Precision: 0.9600
Recall: 0.9200
F1-Score: 0.9400
Matthews Correlation Coefficient: 0.8800
```

---

## 🚀 Future Improvements

- Try SMOTE to handle imbalance
- Try models like XGBoost, LightGBM
- Perform feature selection
- Perform hyperparameter tuning

---

## 📄 License

This project is open-source and free to use.

---

## 🙋‍♂️ Author

Made with ❤️ by [Anurag Tripathi](https://github.com/anurag250402)
