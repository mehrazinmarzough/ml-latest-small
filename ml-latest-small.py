import math
import numpy as np
import pandas as pd

# Load data from CSV files
MoviesDf = pd.read_csv("movies.csv")
RatingsDf = pd.read_csv("ratings.csv")

def CreateMatrix():
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

    sigmas = np.sqrt(Landas)
    Sigma = np.diag(sigmas)

    U = np.empty((m, r))

    for i in range(r):
        U[:, i] = np.matmul(S, V[:, i]) / sigmas[i]

    print(U.shape)
    print(Sigma.shape)
    print(VT.shape)
    X = np.matmul(U, Sigma)
    X = np.matmul(X, VT)  # S = U * Sigma * V
    print(S)
    print(X)

    return X


UserMovieRating = CreateMatrix()
predicted_ratings = SVD(UserMovieRating)

# Get user ID from input
user_id_input = int(input("Enter a user ID: "))
# Check if the entered user ID is valid
if user_id_input not in UserMovieRating.index:
    print(f"User ID {user_id_input} not found in the dataset.")
else:
    # Get the index corresponding to the user ID
    user_index = UserMovieRating.index.get_loc(user_id_input)

    # Get predicted ratings for the user
    user_ratings = predicted_ratings[user_index, :]

    # Find indices of movies with highest predicted ratings
    recommended_movie_indices = np.argsort(user_ratings)[::-1][:10]

    # Print recommended movies
    print(f"\nTop 10 Recommended Movies for User {user_id_input} : \n")
    for index in recommended_movie_indices:
        movie_id = UserMovieRating.columns[index] # Get movie ID from column index
        movie_title = MoviesDf[MoviesDf['movieId'] == movie_id]['title'].values[0] # Get movie title from movie ID
        print(f"Title: {movie_title}")