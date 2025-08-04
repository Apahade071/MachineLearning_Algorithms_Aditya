# Random Forest on Wine Dataset
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load data
wine = load_wine()
X, y = wine.data, wine.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
