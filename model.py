import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def train():
    try:
        data = pd.read_csv("data.csv")

        X = data.drop("target", axis=1)
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = DecisionTreeClassifier(
            random_state=42,
            ccp_alpha=0.0
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        print(f"Accuracy: {accuracy}")

        return accuracy

    except FileNotFoundError:
        print("Error: data.csv file not found.")

    except Exception as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    train()
