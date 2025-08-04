# Support Vector Machine on Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = SVC()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
