import math

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


def SVD(S):
    # Initialize matrices
    m, n = S.shape  # m = number of users, n = number of movies

    # S transpose S
    ST = np.transpose(S)
    STS = np.matmul(ST,S)

    # Calculate Eigen Values and Singular Values
    Landas, V = np.linalg.eig(STS)
    r = len(Landas)
    sigmas = np.zeros((r,m))
    for i in range(r):
        if(Landas[i] < 0):
            Landas[i] = -Landas[i]
        sigmas[i] = math.sqrt(Landas[i])
    sigmas = np.transpose(sigmas)
    V = np.transpose(V)



    # return U, sigma, Vt


UserMovieRating = CreateMatrix()
matrix = np.random.rand(2,3)
SVD(matrix)
