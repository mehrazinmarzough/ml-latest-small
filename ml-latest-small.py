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
    STS = np.matmul(ST, S)

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

    return U, Sigma, VT, S


def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    return dot_product / (norm_u * norm_v)


def get_recommendations(user_id):
    UserMovieRating = CreateMatrix()
    U, Sigma, VT, S = SVD(UserMovieRating)

    # Get the user vector
    user_vector = S[user_id]

    # Calculate cosine similarities between the user vector and all movie vectors
    movie_similarities = np.apply_along_axis(cosine_similarity, 1, VT, user_vector)

    # Sort movies by similarity scores and get the top N recommendations
    N = 10  # Adjust the number of recommendations as needed
    recommended_movie_indices = np.argsort(-movie_similarities)[:N]
    recommended_movie_ids = np.array(MoviesDf['movieId'])[recommended_movie_indices]

    recommended_movies = pd.DataFrame()
    for i in range(10):
        recommended_movies = pd.concat([recommended_movies,
                                        MoviesDf.loc[recommended_movie_ids[i]].to_frame().T], ignore_index=True)

    return recommended_movies.drop('movieId', axis=1)


# Get user ID as input
UserId = int(input("Enter user ID: "))

# Get recommendations for the user
recommendations = get_recommendations(UserId)

print("Recommended movies for user", UserId, ":")
print(recommendations)
