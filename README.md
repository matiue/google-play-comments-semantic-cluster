# semantic clustering of Google Play application comments 

## Motivation
I wanted to understand what people are really saying about Duolingo. Am I the only one who struggles to learn with it? How do others feel?

So I started reading comments. There were thousands. Obviously, I don't have time to read them all so I decided to code something that could.

That same concept now powers caveman.cloud, where most conversations and user queries are already embedded. It's also used for clustering form inputs. ( the final vestion may be cosidered a self-supervised-learning approach)
## methodology 
- using python library meant to scrape google play store 
- cleaning the data
- sentences embeddings using `BAAI/bge-m3`.
- applies HDBSCAN clustering
- find the centroid of clusters(farthest from each others)
- find few mid-range comments to show diversity within the cluster


## Workflow Steps:

1.  **`clean_text(text)`**:
    *   **Purpose**: A utility function to preprocess raw review text.
    *   **Details**: It removes emojis, punctuation, and extra whitespace, ensuring cleaner text for embedding and analysis.

2.  **`scrape_app_reviews(app_id, max_reviews_per_country, countries, days_ago)`**:
    *   **Purpose**: Gathers app reviews from the Google Play Store.
    *   **Details**: Takes an `app_id`, optional `countries` list (defaults to 'us'), `max_reviews_per_country` limit, and `days_ago` to filter reviews within a recent timeframe. It uses the `google_play_scraper` library to fetch reviews, then cleans the 'content' column using `clean_text`, and returns a Pandas DataFrame with 'at', 'country', and 'content' columns.

3.  **Sentence Embeddings (using `BAAI/bge-m3`)**:
    *   **Purpose**: Converts cleaned review text into numerical vector representations.
    *   **Details**: The `SentenceTransformer("BAAI/bge-m3")` model is loaded to generate high-quality embeddings. These embeddings capture the semantic meaning of the reviews, which is crucial for effective clustering.

4.  **`cluster_application(app_name, app_df_slice, min_cluster_size, min_samples)`**:
    *   **Purpose**: Groups similar app reviews into clusters.
    *   **Details**: This function takes the DataFrame of reviews, generates normalized embeddings for the 'content' column, and then applies HDBSCAN clustering. HDBSCAN is robust to noise and can discover clusters of varying densities. It returns the DataFrame with a new 'cluster' column and the normalized embeddings.

5.  **`extract_representatives(app_name, app_df_slice, embeddings, output_path)`**:
    *   **Purpose**: Identifies key comments that represent each cluster.
    *   **Details**: For each identified cluster (excluding noise points), it calculates the centroid of the cluster's embeddings. It then finds the comment closest to the centroid (the most representative) and a few mid-range comments to show diversity within the cluster. Results are returned as a DataFrame and can optionally be saved to a CSV file.

6.  **`run_analysis_workflow(app_id, app_name, max_reviews_per_country, days_ago, min_cluster_size, min_samples, output_path)`**:
    *   **Purpose**: Orchestrates the entire analysis process.
    *   **Details**: This is the main function that ties everything together. It calls `scrape_app_reviews`, `cluster_application`, and `extract_representatives` in sequence. It allows for easy configuration of various parameters like `app_id`, `app_name`, `max_reviews_per_country`, `days_ago`, HDBSCAN parameters (`min_cluster_size`, `min_samples`), and an optional `output_path` for saving the representative comments.

## How to Use:

The workflow can be initiated by calling the `run_analysis_workflow` function, typically within an `if __name__ == '__main__':` block. You can customize the parameters as needed:

```python
final_results_df = run_analysis_workflow(
    app_id='com.spotify.music',       # The Google Play Store ID of the app
    app_name='Spotify',               # A user-friendly name for the app
    max_reviews_per_country=500,    # Maximum reviews to scrape per country
    days_ago=30,                    # Scrape reviews from the last N days
    min_cluster_size=20,            # HDBSCAN parameter: minimum size of clusters
    min_samples=5,                  # HDBSCAN parameter: sensitivity to noise
    output_path=None                # Set to a path like '/content/spotify_clusters.csv' to save results
)
```

## Output:

The `run_analysis_workflow` function returns a Pandas DataFrame containing the identified representative comments for each cluster, including the application name, cluster ID, cluster size, the most representative comment, and a few mid-range comments from each cluster.

