import matplotlib.pyplot as plt

def plot_forecast(train, test, preds):
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train["Sales"], label="Train")
    plt.plot(test.index, test["Sales"], label="Test")
    plt.plot(test.index, preds, label="Forecast", linestyle="--")
    plt.legend()
    plt.title("Retail Sales Forecast")
    plt.show()
