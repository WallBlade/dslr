import numpy as np
import pandas as pd
import mplcursors as mpl
import matplotlib.pyplot as plt
import utils_math as um

def min_max_normalize(column, range_min=0, range_max=1):
    min_val = column.min()
    max_val = column.max()

    return [(range_min + (x - min_val) * (range_max - range_min) / (max_val - min_val)) if pd.notna(x) else None for x in column]

def get_score(df, course):
    house_sums = {
        'Course': course,
        'Gryffindor': um.sum_val(df[course][df['Hogwarts House'] == 'Gryffindor']),
        'Hufflepuff': um.sum_val(df[course][df['Hogwarts House'] == 'Hufflepuff']),
        'Ravenclaw': um.sum_val(df[course][df['Hogwarts House'] == 'Ravenclaw']),
        'Slytherin': um.sum_val(df[course][df['Hogwarts House'] == 'Slytherin'])
    }

    return house_sums

def	get_scores(df, houses_scores):
    '''
    Computes both the scores sums for each house in each course and the
    divergence scores for each course.
    
    Parameters:
    df (pandas.DataFrame): A DataFrame containing the cleaned and normalized
    training dataset.
    houses_scores (list): A list that's meant to store all house_sums.
    
    Returns:
    pandas.DataFrame: A DataFrame containing all the means and divergence scores
    for each house in every course.
    '''
    course_divergence = {}
    for course in df.columns[1:]:
        # Calculate the sum of scores for each house
        house_sums = get_score(df, course)

        houses_scores.append(house_sums)
        
        scores_only = [score for house, score in house_sums.items() if house != 'Course']

        # Calculate the mean of all four houses sums
        course_mean = um.mean_val(scores_only)

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
    '''
    Cleans the dataset by removing useless data and normalizing it for later
    calculations.
    
    Parameters:
    data: The raw dataset.
    
    Returns:
    sanitized: A DataFrame containing the cleaned dataset.
    '''
    numeric_cols = data.select_dtypes(include=['number']).columns.drop(['Index'])
    normalized_data = pd.DataFrame()
    for column in numeric_cols:
        column_data = data[column]
        
        # Normalize the column
        normalized_column = min_max_normalize(column_data)
        
        # Add normalized column to the DataFrame
        normalized_data[column] = normalized_column
      
    house_col = data['Hogwarts House']
    sanitized = pd.concat([house_col, normalized_data], axis=1)

    return sanitized

def draw_histograms(divergence, houses_scores):
    '''
    Plots two histograms, first one displays the sums of scores for each house
    in every course. Second one displays the divergence scores in each course.
    
    Parameters:
    divergence: The divergence scores in every course.
    houses_scores: The sums of scores for each house in every course.
    '''
    houses_scores_df = pd.DataFrame(houses_scores).set_index('Course')

    fig, ax = plt.subplots(2, 1, figsize=(18, 10))

    courses = houses_scores_df.index
    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

    x = np.arange(len(courses))
    width = 0.2

    for i, house in enumerate(houses):
        ax[0].bar(x + i * width, houses_scores_df[house], width, label=house)

    # Add labels and title
    ax[0].set_ylabel("Scores")
    ax[0].set_title("Scores des 4 maisons par matière")
    ax[0].set_xticks(x + width * 1.5)
    ax[0].set_xticklabels(courses, rotation=45, ha="right")

    ax[0].legend(title="Maison")
    
    # Extract course titles (keys of the divergence dictionary)
    courses = divergence.columns
    divergences = divergence.iloc[1]

    # Plot the histogram
    bars = ax[1].bar(courses, divergences)

    # Add labels and title
    ax[1].set_xlabel("Courses")
    ax[1].set_ylabel("Divergence")
    ax[1].set_title("Scores de divergence par matière")

    # Rotate x-axis labels if there are many courses for better readability
    plt.xticks(rotation=45, ha="right")

    # Add hover functionality
    cursor = mpl.cursor(bars, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set(
        text=f"Divergence: {divergences.iloc[sel.index]:.2f}",
        bbox=dict(facecolor="white", edgecolor="black")
    ))

    # Rearrange layout for clean display
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('datasets/dataset_train.csv')
    df = clean_data(data)
    houses_scores = []
    divergence = get_scores(df, houses_scores)
    draw_histograms(divergence, houses_scores)