import pandas as pd
import numpy as np


def read_file(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print("Произошла ошибка при чтении файла:", filepath)
        raise e
    return df


# по сути то же самое, что df.describe, но только с нужными по условию признаками
def get_statistic_description(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns

    counts = {}
    mins = {}
    maxs = {}
    means = {}

    df.columns = np.arange(len(cols))

    backward_transform = {}

    for i in range(len(cols)):
        backward_transform[df.columns[i]] = cols[i]

    for col in df.columns:
        counts[col] = df[col].count()
        mins[col] = df[col].min()
        maxs[col] = df[col].max()
        means[col] = df[col].mean()

    qualities = [
        ('count', counts),
        ('min', mins),
        ('max', maxs),
        ('mean', means),
    ]

    index = [quality[0] for quality in qualities]
    data = [quality[1] for quality in qualities]

    description = pd.DataFrame(data, index=index)
    description.rename(columns=backward_transform, inplace=True)

    return description


class LinearRegressionModel:
    def __init__(self) -> None:
        self.w = 0
        self.b = 0

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=1000, learning_rate=0.001) -> None:
        for i in range(epochs):
            y_pred = self.predict(x)

            self.backward_pass(x, y, y_pred, learning_rate)

    def predict(self, x: np.ndarray) -> np.ndarray:
        y_pred = self.w * x + self.b
        return y_pred

    def backward_pass(self, x: np.ndarray, y: np.ndarray, y_pred, learning_rate) -> None:
        n = x.shape[0]

        if n == 0:
            return

        grad_w = (2/n) * np.sum((y_pred - y)*x)
        grad_b = (2/n) * np.sum((y_pred - y))

        self.w -= learning_rate * grad_w
        self.b -= learning_rate * grad_b
