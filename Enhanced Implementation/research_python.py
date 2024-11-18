# %% [markdown]
# # Front Matter

# %%
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import accuracy
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error

# %% [markdown]
# # 1. Data Preparation
# ## Load and Clean Data

# %%
# Load datasets
movies = pd.read_csv(r'C:\Users\pedro\Desktop\Github\DS340-Midterm\Small MovieLens\movies.csv')
ratings = pd.read_csv(r'C:\Users\pedro\Desktop\Github\DS340-Midterm\Small MovieLens\ratings.csv')
tags = pd.read_csv(r'C:\Users\pedro\Desktop\Github\DS340-Midterm\Small MovieLens\tags.csv')

# %%
# Display first few rows
print("Movies DataFrame:")
display(movies.head())

print("Ratings DataFrame:")
display(ratings.head())

print("Tags DataFrame:")
display(tags.head())

# %%
# Select necessary columns
movies = movies[['movieId', 'title', 'genres']]
ratings = ratings[['userId', 'movieId', 'rating']]
tags = tags[['movieId', 'tag']]

# %%
validate_indices = ratings.sample(frac=0.1, random_state=42).index
validate_set = ratings.loc[validate_indices]

# Remaining data for training/testing
remaining_data = ratings.drop(index=validate_indices)

# Perform 80/20 split on the remaining data
train_set, test_set = sk_train_test_split(remaining_data, test_size=0.2, random_state=42)

print(f"Validation Set: {len(validate_set)} entries")
print(f"Training Set: {len(train_set)} entries")
print(f"Testing Set: {len(test_set)} entries")

# %%
# Merge Movies and Ratings to create CF DataFrame
cf = pd.merge(movies, ratings, on='movieId', how='inner')

# Merge Movies and Tags to create CBF DataFrame
cbf = pd.merge(movies, tags, on='movieId', how='inner')

# %%
# Check data types
print("\nData types in CF DataFrame:")
print(cf.dtypes)

print("\nData types in CBF DataFrame:")
print(cbf.dtypes)

# %%
# Function to extract year from title
def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    else:
        return np.nan  # Handle cases where year is not found

# Apply the function to create a 'year' column
movies['year'] = movies['title'].apply(extract_year)

# Clean the 'title' by removing the year and converting to lowercase
movies['title_clean'] = movies['title'].apply(lambda x: re.sub(r'\s*\(\d{4}\)', '', x).lower())

# Verify the changes
print("Sample of Movies DataFrame after extracting year and cleaning title:")
display(movies[['movieId', 'title', 'title_clean', 'year']].head())

# %%
# Group tags by 'movieId' and concatenate them into a single string
tags_grouped = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

# Verify the grouped tags
print("\nSample of Grouped Tags DataFrame:")
display(tags_grouped.head())

# %%
# Merge 'tags_grouped' with 'movies' DataFrame to update 'cbf'
cbf = pd.merge(movies, tags_grouped, on='movieId', how='left')

# Replace NaN tags with empty strings (for movies without tags)
cbf['tag'] = cbf['tag'].fillna('')

# Verify the merged DataFrame
print("\nSample of Content-Based Filtering (CBF) DataFrame after merging tags:")
display(cbf.head())

# %%
# Create 'year_str' column for concatenation
cbf['year_str'] = cbf['year'].astype(str)

# Combine genres, title_clean, tags, and year into the 'related' column
cbf['related'] = cbf['genres'].str.replace('|', ' ') + ' ' + cbf['title_clean'] + ' ' + cbf['tag'] + ' ' + cbf['year_str']

# Verify the 'related' column
print("\nSample of CBF DataFrame with 'related' column:")
display(cbf[['movieId', 'title', 'related']].head())

# %%
# Preprocess the 'related' column
cbf['related'] = cbf['related'].str.lower()  # Ensure lowercase
cbf['related'] = cbf['related'].str.replace(r'\d+', '', regex=True)  # Remove numbers
cbf['related'] = cbf['related'].str.replace(r'[^a-z\s]', '', regex=True)  # Remove special characters
cbf['related'] = cbf['related'].str.strip()  # Remove extra spaces

# Verify the preprocessing
print("\nSample of CBF DataFrame after preprocessing 'related' column:")
display(cbf[['movieId', 'title', 'related']].head())

# %%
# Check for NaN values in 'related' column
nan_related = cbf['related'].isna().sum()
print(f"\nNumber of NaN values in 'related' column: {nan_related}")

# %%
# Save CF and CBF DataFrames to CSV (optional)
# Uncomment if you need to save for later use
cf.to_csv(r'C:\Users\pedro\Desktop\Github\DS340-Midterm\Small MovieLens\cf.csv', index=False)
cbf.to_csv(r'C:\Users\pedro\Desktop\Github\DS340-Midterm\Small MovieLens\cbf.csv', index=False)

print("\nCleaned CF and CBF DataFrames are ready.")

# %% [markdown]
# # Content-Based

# %% [markdown]
# ## Initiating TF-IDF

# %%
# Initialize TF-IDF Vectorizer with English stop words
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the 'related' column
tfidf_matrix = tfidf.fit_transform(cbf['related'])

# Verify the shape of the TF-IDF matrix
print(f"\nTF-IDF Matrix Shape: {tfidf_matrix.shape}")

# %% [markdown]
# ## Cosine Similarity

# %%
# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Display the similarity matrix shape (should be a square matrix: number of movies x number of movies)
print(cosine_sim.shape)

# %%
# Create a function that takes in a movie title and gives recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Convert input title to lowercase
    title_cleaned = title.lower()

    # Find the index of the movie in the 'title_clean' column
    idx = cbf[cbf['title_clean'] == title_cleaned].index[0]

    # Get the pairwise similarity scores for all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the 10 most similar movies
    sim_scores = sim_scores[1:11]  # Skip the first movie (itself)

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies
    return cbf['title'].iloc[movie_indices]

# Example: Get recommendations for a movie

recommendations = get_recommendations('copycat')
print(recommendations)


# %% [markdown]
# # Collaborative

# %%
# Prepare the dataset (from your CF data)
# We'll use only the necessary columns
cf_reduced = cf[['userId', 'movieId', 'rating']]

# Define the format for the dataset using Reader
reader = Reader(rating_scale=(0.5, 5))  # Adjust the rating scale if necessary

# Load the data into Surprise format
data = Dataset.load_from_df(cf_reduced, reader)

# Train-test split (80% train, 20% test)
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize the SVD model for matrix factorization
svd = SVD()

# Train the model on the training set
svd.fit(trainset)

# Test the model on the test set and evaluate performance (RMSE)
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)

# %%
# Making movie recommendations for a specific user
def recommend_movies(user_id, model=svd, n_recommendations=10):
    # Get a list of all movie IDs
    all_movie_ids = cf['movieId'].unique()

    # Predict ratings for all movies the user hasn't rated yet
    user_rated_movies = cf[cf['userId'] == user_id]['movieId'].tolist()
    unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in user_rated_movies]

    # Predict ratings for unrated movies
    predictions = [model.predict(user_id, movie_id) for movie_id in unrated_movies]

    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    top_n = predictions[:n_recommendations]

    # Return movie IDs and estimated ratings
    recommended_movies = [(cf[cf['movieId'] == pred.iid]['title'].values[0], pred.est) for pred in top_n]

    return recommended_movies

# Example: Get recommendations for user 1
recommended_movies = recommend_movies(user_id=1)
print(recommended_movies)

# %%
cross_val_results = cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# %%
# Display the average RMSE and MAE
print("\nCross-Validation Results:")
print(f"Average RMSE: {cross_val_results['test_rmse'].mean():.4f}")
print(f"Average MAE: {cross_val_results['test_mae'].mean():.4f}")

# %%
# Define a parameter grid for SVD
param_grid = {
    'n_factors': [50, 100, 150],  # Number of latent factors
    'n_epochs': [20, 30],         # Number of training epochs
    'lr_all': [0.002, 0.005],     # Learning rate for all parameters
    'reg_all': [0.02, 0.05]       # Regularization term for all parameters
}

# Initialize GridSearchCV with SVD algorithm
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, joblib_verbose=1)

# Perform grid search
print("\nStarting Grid Search for hyperparameter tuning...")
gs.fit(data)
print("Grid Search completed.")

# Extract the best RMSE score
print(f"\nBest RMSE Score: {gs.best_score['rmse']:.4f}")

# Extract the best parameters
print("Best parameters:")
print(gs.best_params['rmse'])

# %%
# Train the optimal model with best parameters
best_params = gs.best_params['rmse']
optimal_svd = SVD(
    n_factors=best_params['n_factors'],
    n_epochs=best_params['n_epochs'],
    lr_all=best_params['lr_all'],
    reg_all=best_params['reg_all']
)

print("\nTraining the optimized SVD model with best parameters...")
optimal_svd.fit(trainset)
print("Optimized model training completed.")

# %%
# Predict on the test set using the optimized model
print("\nMaking predictions on the test set with the optimized model...")
optimal_predictions = optimal_svd.test(testset)

# Compute RMSE for the optimized model
optimal_rmse = accuracy.rmse(optimal_predictions)
print(f"Optimal Collaborative Filtering RMSE: {optimal_rmse:.4f}")

# %%
recommended_movies = recommend_movies(user_id=1, model = optimal_svd)
print(recommended_movies)

# %% [markdown]
# # Hybrid System

# %%
# Function to predict content-based rating for a given user and movie
def predict_cbf_rating(movie_id, cosine_sim_matrix, cbf_df, target_movie_index, user_movies, n_similar=10):
    # Get cosine similarity scores for the target movie
    sim_scores = list(enumerate(cosine_sim_matrix[target_movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the top similar movies
    sim_scores = sim_scores[1:n_similar + 1]
    
    # Calculate the weighted sum of user ratings for similar movies
    weighted_sum = 0
    sim_sum = 0
    for idx, sim_score in sim_scores:
        movie_id_similar = cbf_df.iloc[idx]['movieId']
        if movie_id_similar in user_movies:
            weighted_sum += sim_score * user_movies[movie_id_similar]
            sim_sum += sim_score
    
    # Return the weighted average rating
    if sim_sum == 0:
        return np.mean(list(user_movies.values()))  # Default to user's average rating if no similar movies
    return weighted_sum / sim_sum


# %%
# Hybrid recommendation: Combine CF and CBF
def hybrid_recommendation(user_id, movie_id, cf_model, cosine_sim_matrix, cbf_df, cf_df, weight_cf=0.7, weight_cbf=0.3):
    # Get the predicted CF rating (Collaborative Filtering)
    cf_prediction = cf_model.predict(user_id, movie_id).est
    
    # Get the predicted CBF rating (Content-Based Filtering)
    # First, get the movie index in CBF
    target_movie_index = cbf_df[cbf_df['movieId'] == movie_id].index[0]
    
    # Get the movies the user has rated
    user_ratings = cf_df[cf_df['userId'] == user_id].set_index('movieId')['rating'].to_dict()
    
    # Get the CBF rating prediction
    cbf_prediction = predict_cbf_rating(movie_id, cosine_sim_matrix, cbf_df, target_movie_index, user_ratings)
    
    # Combine the predictions using the weights
    final_rating = (weight_cf * cf_prediction) + (weight_cbf * cbf_prediction)
    
    return final_rating


# %%
def recommend_hybrid_movies(user_id, cf_model, cosine_sim_matrix, cbf_df, cf_df, n_recommendations=10, weight_cf=0.7, weight_cbf=0.3):
    all_movie_ids = cf_df['movieId'].unique()
    user_rated_movies = cf_df[cf_df['userId'] == user_id]['movieId'].tolist()
    unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in user_rated_movies]
    
    predictions = []
    
    for movie_id in unrated_movies:
        # Predict the hybrid rating
        predicted_rating = hybrid_recommendation(user_id, movie_id, cf_model, cosine_sim_matrix, cbf_df, cf_df, weight_cf, weight_cbf)
        predictions.append((movie_id, predicted_rating))
    
    # Sort the movies by the predicted hybrid rating and get the top N
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_predictions = predictions[:n_recommendations]
    
    # Return the top N recommended movie titles
    return [(cbf_df[cbf_df['movieId'] == movie_id]['title'].values[0], rating) for movie_id, rating in top_n_predictions]




# %%
# Example: Get hybrid recommendations for user 1
recommendations = recommend_hybrid_movies(user_id=1, cf_model=optimal_svd, cosine_sim_matrix=cosine_sim, cbf_df=cbf, cf_df=cf)
recommendations

# %% [markdown]
# # Validation

# %%
weight_cf = 0.4
weight_cbf = 1 - weight_cf

# %%
def generate_hybrid_predictions(testset, cf_model, cosine_sim_matrix, cbf_df, cf_df, weight_cf, weight_cbf):
    predictions = []
    for user_id, movie_id, true_rating in testset:
        # Predict the hybrid rating
        pred_rating = hybrid_recommendation(user_id, movie_id, cf_model, cosine_sim_matrix, cbf_df, cf_df, weight_cf, weight_cbf)
        predictions.append((user_id, movie_id, true_rating, pred_rating))
    return predictions
# Example: Generating predictions for a test set
hybrid_predictions = generate_hybrid_predictions(testset, optimal_svd, cosine_sim, cbf, cf, weight_cf, weight_cbf)


# %% [markdown]
# # RMSE and MAE

# %%

# Modified HybridPrediction class to mimic surprise's Prediction object
class HybridPrediction:
    def __init__(self, uid, iid, r_ui, est):
        self.uid = uid  # User ID
        self.iid = iid  # Movie ID
        self.r_ui = r_ui  # True rating
        self.est = est  # Predicted rating
        self.details = {}  # Can be empty, but required by surprise's accuracy functions

    # Making the object iterable like Surprise's Prediction class
    def __iter__(self):
        return iter((self.uid, self.iid, self.r_ui, self.est, self.details))
    
def calculate_rmse_mae(hybrid_predictions):
    # Create list of HybridPrediction objects
    surprise_predictions = [HybridPrediction(uid, iid, r_ui, est) for (uid, iid, r_ui, est) in hybrid_predictions]

    # RMSE
    rmse = accuracy.rmse(surprise_predictions, verbose=False)

    # MAE
    mae = accuracy.mae(surprise_predictions, verbose=False)

    return rmse, mae

# %%
rmse, mae = calculate_rmse_mae(hybrid_predictions)


# %% [markdown]
# # F1-Score, Precision, Recall

# %%
# Step 1: Prepare y_true and y_pred for binary classification
threshold = 4.5  # Set threshold for relevance

# Generate binary labels based on true ratings and predicted ratings

# Step 2: Calculate Precision, Recall, and F1-Score
def calculate_precision_recall_f1(threshold, predictions): #
    y_true = [int(true_rating >= threshold) for (_, _, true_rating, _) in predictions]  # Actual ratings
    y_pred = [int(pred_rating >= threshold) for (_, _, _, pred_rating) in predictions]  # Predicted ratings
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return precision, recall, f1
    
precision, recall, f1 = calculate_precision_recall_f1(threshold, hybrid_predictions)

# %%
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")


