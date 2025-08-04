# Decision Tree on Titanic Dataset (sample from seaborn)
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'embarked', 'class', 'sex', 'survived'])

# Preprocess
X = df[['age', 'fare']]
X['sex'] = df['sex'].map({'male': 0, 'female': 1})
y = df['survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
