from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import src.constants as const
from src.utils import load_dataset, Standardizer, Evaluation

X, y = load_dataset(const.DATASET_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, shuffle=True)

scaler = Standardizer(columns_to_standardize=const.NUMERIC_COLUMNS)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = LogisticRegression(dual=False)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

results = Evaluation(actuals=y_test, predictions=y_pred)
results.print()
