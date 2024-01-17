import numpy as np
import pandas as pd
def CreateMatrix():
    # Load data from CSV files
    MoviesDf = pd.read_csv("movies.csv")
    RatingsDf = pd.read_csv("ratings.csv")

    user_movie_ratings = pd.merge(RatingsDf, MoviesDf, on='movieId')[['userId', 'movieId', 'rating']]
    user_movie_ratings_pivot = user_movie_ratings.pivot_table(index='userId', columns='movieId', values='rating')
    user_movie_ratings_matrix = user_movie_ratings_pivot.fillna(0).values

    return user_movie_ratings_matrix


UserMovieRating = CreateMatrix()

