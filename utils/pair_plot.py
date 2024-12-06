import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

def get_colors(houses):
    """
    Get the colors for the given houses

    Parameters: houses (DataFrame of the houses to get the colors for)
    Returns the colors for the given houses
    """

    colors = []
    for house in houses:
        if house == 'Ravenclaw':
            colors.append('magenta')
        elif house == 'Slytherin':
            colors.append('cyan')
        elif house == 'Gryffindor':
            colors.append('black')
        elif house == 'Hufflepuff':
            colors.append('orange')
    return colors

def display_KDE_graph(ax, data):
    """
    Display a KDE graph for the given data

    Arguments: ax (the axis to display the graph), data (the data to display, must contains 'Hogwarts House' and the feature to display)
    """

    ravenclaw_values = data[data['Hogwarts House'] == 'Ravenclaw'].iloc[:, 1].to_numpy()
    slytherin_values = data[data['Hogwarts House'] == 'Slytherin'].iloc[:, 1].to_numpy()
    gryffindor_values = data[data['Hogwarts House'] == 'Gryffindor'].iloc[:, 1].to_numpy()
    hufflepuff_values = data[data['Hogwarts House'] == 'Hufflepuff'].iloc[:, 1].to_numpy()

    sns.kdeplot(ravenclaw_values, fill=True, color="magenta", legend=False, ax=ax)
    sns.kdeplot(slytherin_values, fill=True, color="cyan", legend=False, ax=ax)
    sns.kdeplot(gryffindor_values, fill=True, color="black", legend=False, ax=ax)
    sns.kdeplot(hufflepuff_values, fill=True, color="orange", legend=False, ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def display_pair_graph(ax, data):
    """
    Display a KDE graph for the given data

    Arguments: ax (the axis to display the graph), data (the data to display, must contains 'Hogwarts House' and the feature to display)
    """

    data.dropna()
    x_values = data.iloc[:, 1]
    y_values = data.iloc[:, 2]
    colors = get_colors(data['Hogwarts House'])
    ax.scatter(x_values, y_values, c=colors, alpha=0.5, s=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def set_axis_settings(ax, feature_x, feature_y, x, y):
    if x == 0: ax.set_ylabel(feature_y[:4])
    else : ax.set_ylabel('')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(feature_x[:4])

def main():
    try:
        # --- Retrieve dataset and setup basic variables --- #
        np.set_printoptions(suppress=True)
        data = pd.read_csv("datasets/dataset_train.csv")
        data = data.drop('Defense Against the Dark Arts', axis=1)
        size = 12

        # --- Display the pair plot --- #
        fig, axs = plt.subplots(size, size, figsize=(16, 10))
        y = 0
        x = 0

        # --- Fill the axis with the pair/KDE graph --- #
        for feature_y in data.loc[:, 'Arithmancy':'Flying']:
            for feature_x in data.loc[:, 'Arithmancy':'Flying']:
                if feature_x == feature_y:
                    display_KDE_graph(axs[y, x], data[['Hogwarts House', feature_x]])
                else:
                    display_pair_graph(axs[y, x], data[['Hogwarts House', feature_x, feature_y]])
                set_axis_settings(axs[y, x], feature_x, feature_y, x, y)
                x += 1
            x = 0
            y += 1

        # --- Display graphs in fullscreen --- #
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1, wspace=0.1)
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.show()
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")

if __name__ == "__main__":
    main()