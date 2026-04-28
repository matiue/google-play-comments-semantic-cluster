from sklearn.preprocessing import normalize
import hdbscan
import numpy as np
import re
import string
import pandas as pd
from google_play_scraper import Sort, reviews
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove emojis
    emoji_pattern = re.compile("[" # Start of pattern
                               "\U0001F600-\U0001F64F"  # emoticons
                               "\U0001F300-\U0001F5FF"  # symbols & pictographs
                               "\U0001F680-\U0001F6FF"  # transport & map symbols
                               "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "\U00002702-\U000027B0"  # Dingbats
                               "\U000024C2-\U0001F251"  # Enclosed characters
                               "\U0001f926-\U0001f937"
                               "\U0001f91d-\U0001f93a"
                               "\U0000200D"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Faster alternative to regex for basic punctuation removal
    # This keeps all alphanumeric characters and removes punctuation
    table = str.maketrans(string.punctuation.replace('_', ''), ' ' * (len(string.punctuation) - 1))
    text = text.translate(table)

    # Explicitly remove underscores
    text = text.replace('_', ' ')

    # Use regex only for whitespace collapsing
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def scrape_app_reviews(app_id, max_reviews_per_country=2000, countries=None, days_ago=30):
    """
    Scrapes user reviews for a given app ID from specified countries,
    getting as many reviews as possible up to a limit per country using pagination.
    Reviews are limited to the last `days_ago`.
    """
    all_reviews_data = []

    if countries is None:
        countries_to_scrape = ['us']
        print("No specific countries provided. Defaulting to 'us'.")
    else:
        countries_to_scrape = countries

    # Diagnostic prints
    current_time = datetime.now()
    print(f"[Diagnostic] Current datetime.now(): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Diagnostic] days_ago parameter received: {days_ago}")

    # Calculate the cutoff date for reviews
    cutoff_date = current_time - timedelta(days=days_ago)
    print(f"Fetching reviews from the last {days_ago} days (after {cutoff_date.strftime('%Y-%m-%d')})")

    for country_code in countries_to_scrape:
        print(f"Scraping reviews for app ID: {app_id} in country: {country_code}")
        current_reviews = []
        continuation_token = None
        reviews_count = 0
        page_num = 0
        date_limit_reached = False

        while reviews_count < max_reviews_per_country and not date_limit_reached:
            page_num += 1
            print(f"  Fetching page {page_num} for {country_code} (total scraped: {reviews_count})")

            result, continuation_token = reviews(
                app_id,
                lang='en',
                country=country_code,
                sort=Sort.NEWEST,
                count=200,
                continuation_token=continuation_token
            )

            if not result:
                print(f"  No more reviews found for {country_code}.")
                break

            for review in result:
                # Check if the review date is within the last `days_ago`
                if review['at'].replace(tzinfo=None) >= cutoff_date:
                    review['country'] = country_code
                    current_reviews.append(review)
                else:
                    # If we find a review older than the cutoff, stop for this country
                    date_limit_reached = True
                    break

            reviews_count = len(current_reviews)

            if continuation_token is None or date_limit_reached:
                print(f"  End of pages or date limit reached for {country_code}.")
                break

        print(f"  Finished scraping {len(current_reviews)} reviews from {country_code}")
        all_reviews_data.extend(current_reviews)

    df_reviews = pd.DataFrame(all_reviews_data)

    # Pandas Optimizations for DataFrame types
    if not df_reviews.empty:
        # Ensure 'at' is a proper datetime object without timezone for consistency
        df_reviews['at'] = pd.to_datetime(df_reviews['at']).dt.tz_localize(None)
        # Ensure text columns are string type
        df_reviews['content'] = df_reviews['content'].astype(str)
        df_reviews['userName'] = df_reviews['userName'].astype(str)

    # Select only the requested columns and reassign to modify the DataFrame in place.
    # Ensure 'score' is included here
    if not df_reviews.empty:
        df_reviews = df_reviews[['at', 'country', 'content']]

        # Drop rows with any NaN values in the selected columns
        df_reviews.dropna(inplace=True)

        # Apply the existing clean_text function directly to the 'content' column
        df_reviews['content'] = df_reviews['content'].apply(clean_text)
    return df_reviews


def cluster_application(app_name, app_df_slice, min_cluster_size=20, min_samples=5):
    """
    Embeds and clusters comments for a single application.

    Uses normalized embeddings + Euclidean distance to emulate cosine similarity.

    Returns:
        app_df_slice: DataFrame with a new 'cluster' column
        embeddings_norm: normalized embeddings used for clustering
    """
    comments = app_df_slice["content"].tolist()
    print(f"\nProcessing embeddings & clustering for: {app_name} | {len(comments)} comments")

    # Step 1: Generate embeddings
    embeddings = model.encode(
        comments,
        batch_size=64,   # adjust for your memory
        show_progress_bar=True,
        normalize_embeddings=False  # do NOT normalize yet
    )
    print ("embeding done")
    # Step 2: Normalize embeddings (for cosine similarity)
    embeddings_norm = normalize(embeddings)
    print ("normalize done")

    # Step 3: HDBSCAN clustering (Euclidean on normalized vectors ≈ cosine similarity)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean"
    )
    print ("hdbscan done")

    labels = clusterer.fit_predict(embeddings_norm)

    # Attach clusters to the DataFrame slice
    app_df_slice = app_df_slice.copy()
    app_df_slice["cluster"] = labels

    print(f"Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)} | Noise points: {(labels == -1).sum()}")
    return app_df_slice, embeddings_norm

def extract_representatives(app_name, app_df_slice, embeddings, output_path=None):
    """
    Extract representative comments per cluster:
        - Closest to centroid
        - Mid-range for diversity
    """
    cluster_results = []

    for cluster_id in sorted(set(app_df_slice["cluster"])):
        if cluster_id == -1:
            continue  # skip noise

        cluster_mask = app_df_slice["cluster"] == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_comments = app_df_slice[cluster_mask]

        # Compute centroid
        centroid = cluster_embeddings.mean(axis=0)

        # Cosine distance to centroid
        dot = np.dot(cluster_embeddings, centroid)
        norm_a = np.linalg.norm(cluster_embeddings, axis=1)
        norm_b = np.linalg.norm(centroid)
        cosine_sim = dot / (norm_a * norm_b + 1e-9)
        distances = 1 - cosine_sim

        # Closest comment to centroid
        best_idx = np.argmin(distances)
        representative = cluster_comments.iloc[best_idx]

        # Mid-range comments (30–70 percentile)
        mid_idx = np.where((distances > np.percentile(distances, 30)) &
                           (distances < np.percentile(distances, 70)))[0]
        mid_comments = [cluster_comments.iloc[idx]["content"] for idx in
                        np.random.choice(mid_idx, min(3, len(mid_idx)), replace=False)]

        cluster_results.append({
            "application": app_name,
            "cluster_id": cluster_id,
            "cluster_size": len(cluster_comments),
            "representative_comment": representative["content"],
            "mid_range_comments": mid_comments
        })

    result_df = pd.DataFrame(cluster_results)

    if output_path: # Save only if output_path is provided
        result_df.to_csv(output_path, index=False)
        print(f"Saved representative comments for {app_name} -> {output_path}")

    return result_df

def run_analysis_workflow(app_id='com.spotify.music', app_name='Spotify', max_reviews_per_country=500, days_ago=30, min_cluster_size=20, min_samples=5, output_path=None):
    print(f"Starting analysis for app: {app_name} (ID: {app_id})")

    # Step 1: Scrape reviews
    print("Scraping reviews...")
    scrapped_reviews_df = scrape_app_reviews(app_id, max_reviews_per_country=max_reviews_per_country, days_ago=days_ago)

    if scrapped_reviews_df.empty:
        print("No reviews scraped. Exiting workflow.")
        return pd.DataFrame()

    print(f"Total reviews scraped: {len(scrapped_reviews_df)}")
    display(scrapped_reviews_df.head())

    # Step 2: Cluster the scraped reviews
    print("Clustering reviews...")
    spotify_clustered_df, spotify_embeddings_norm = cluster_application(app_name, scrapped_reviews_df, min_cluster_size=min_cluster_size, min_samples=min_samples)
    display(spotify_clustered_df.head())

    # Step 3: Extract representatives from the clustered reviews
    print("Extracting representative comments...")
    spotify_reps_df = extract_representatives(app_name, spotify_clustered_df, spotify_embeddings_norm, output_path=output_path)
    display(spotify_reps_df.head())
    return spotify_reps_df

if __name__ == '__main__':
    # Example usage when run as a script
    final_results_df = run_analysis_workflow(
        app_id='com.spotify.music',
        app_name='Spotify',
        max_reviews_per_country=500,
        days_ago=30,
        min_cluster_size=20,
        min_samples=5,
        output_path=None # Set to a path like '/content/spotify_clusters.csv' to save results
    )
    print("Workflow completed. Final representative comments:")
    display(final_results_df.head())