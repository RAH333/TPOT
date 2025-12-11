# 1. Installing TPOT
#pip install tpot
# 2. Importing Libraries
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 3. Loading and Splitting Data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# 4. Initializing TPOT
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    random_state=42
)
# 5. Training the Model
tpot.fit(X_train, y_train)
# 6. Evaluating Accuracy
y_pred = tpot.fitted_pipeline_.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
# 7. Exporting the Best Pipeline
from joblib import dump

dump(tpot.fitted_pipeline_, "best_pipeline.pkl")
print("Pipeline saved as best_pipeline.pkl")

#You can load it later as follows:

from joblib import load

model = load("best_pipeline.pkl")
predictions = model.predict(X_test)
