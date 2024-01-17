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

    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(STS)

    # Filter out negative eigenvalues and their corresponding eigenvectors
    Landas = eigenvalues[eigenvalues >= 0]
    V = eigenvectors[:, eigenvalues >= 0]
    VT = np.transpose(V)

    r = len(Landas)
    sigmas = Landas
    for i in range(r):
        sigmas[i] = math.sqrt(Landas[i])

    
    # return U, sigma, Vt


UserMovieRating = CreateMatrix()
matrix = np.random.rand(2,3)
SVD(matrix)
