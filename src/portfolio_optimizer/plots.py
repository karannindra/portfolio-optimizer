import matplotlib.pyplot as plt


def plot_weights(weights, asset_names, title):

    plt.figure(figsize=(10, 6))

    plt.bar(asset_names, weights)

    plt.xticks(rotation=45)

    plt.ylabel("Portfolio Weight")

    plt.title(title)

    plt.grid(True)

    plt.show()