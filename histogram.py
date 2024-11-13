import numpy as np
import pandas as pd
import mplcursors as mpl
import matplotlib.pyplot as plt
from utils import utils_math as mt

def min_max_normalize(column, range_min=0, range_max=1):
    # Calcul des valeurs minimum et maximum dans la colonne
    min_val = column.min()
    max_val = column.max()
    
    # Normalisation min-max sur une échelle entre range_min et range_max
    return [(range_min + (x - min_val) * (range_max - range_min) / (max_val - min_val)) if pd.notna(x) else None for x in column]

def	get_score(df, houses_scores):
    course_divergence = {}
    for course in df.columns[1:]:
        # Calculate the sum of scores for each house
        house_sums = {
            'Course': course,
            'Gryffindor': mt.sum_val(df[course][df['Hogwarts House'] == 'Gryffindor']),
            'Hufflepuff': mt.sum_val(df[course][df['Hogwarts House'] == 'Hufflepuff']),
            'Ravenclaw': mt.sum_val(df[course][df['Hogwarts House'] == 'Ravenclaw']),
            'Slytherin': mt.sum_val(df[course][df['Hogwarts House'] == 'Slytherin'])
        }

        houses_scores.append(house_sums)
        
        scores_only = [score for house, score in house_sums.items() if house != 'Course']

        # Calculate the mean of all four houses sums
        course_mean = mt.mean_val(scores_only)

        # Calculate the divergence score for [course]
        divergence = 0
        for score in scores_only:
            divergence += (score - course_mean) ** 2
        divergence /= 4

        course_divergence[course] = {
            'mean': course_mean,
            'divergence': divergence
        }

    return pd.DataFrame(course_divergence)

def clean_data(data):
    numeric_cols = data.select_dtypes(include=['number']).columns.drop(['Index'])
    normalized_data = pd.DataFrame()
    for column in numeric_cols:
        column_data = data[column]
        mean = mt.mean_val(column_data)
        std = mt.std_val(column_data)
        
        # Normalize the column
        normalized_column = min_max_normalize(column_data)
        
        # Add normalized column to the DataFrame
        normalized_data[column] = normalized_column
      
    house_col = data['Hogwarts House']
    sanitized = pd.concat([house_col, normalized_data], axis=1)

    return sanitized

def draw_histogram_by_course(houses_scores):
    houses_scores_df = pd.DataFrame(houses_scores).set_index('Course')

    fig, ax = plt.subplots(figsize=(14, 8))

    courses = houses_scores_df.index
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    x = np.arange(len(courses))
    width = 0.2

    for i, house in enumerate(houses):
        ax.bar(x + i * width, houses_scores_df[house], width, label=house)

    ax.set_xlabel("Courses")
    ax.set_ylabel("Scores")
    ax.set_title("Scores des 4 maisons par matière")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(courses, rotation=45, ha="right")

    ax.legend(title="Maison")

    plt.show()

def draw_histogram(divergence, houses_scores):
    fig, ax = plt.subplots(figsize=(14, 8))

    # Extract course titles (keys of the divergence dictionary)
    courses = divergence.columns
    divergences = divergence.iloc[1]

    # Plot the histogram
    bars = ax.bar(courses, divergences)


    # Add labels and title
    ax.set_xlabel("Courses")
    ax.set_ylabel("Divergence")
    ax.set_title("Divergence Score per Course")

    # Rotate x-axis labels if there are many courses for better readability
    plt.xticks(rotation=45, ha="right")

    # Add hover functionality
    cursor = mpl.cursor(bars, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set(
        text=f"Divergence: {divergences.iloc[sel.index]:.2f}",
        bbox=dict(facecolor="white", edgecolor="black")
    ))

    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('datasets/dataset_train.csv')
    df = clean_data(data)
    houses_scores = []
    divergence = get_score(df, houses_scores)
    print(pd.DataFrame(houses_scores))
    draw_histogram_by_course(houses_scores)