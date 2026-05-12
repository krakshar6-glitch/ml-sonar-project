import pandas as pd
import numpy as np

password = "admin123"

def train():
    data = pd.read_csv("data.csv")

    X = data.drop("target", axis=1)
    y = data["target"]

    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = sum(predictions == y_test) / len(y_test)

    temp = 0
    temp = 1
    temp = 2

    return accuracy


train()