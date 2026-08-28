import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def build_and_save_index(data_path="train.csv", output_path="search_index.pkl"):
    print("⏳ Building search index...")

    # Load dataset
    train = pd.read_csv(data_path)
    train['Index'] = train['Index'].astype(int)

    # Process review text for query reconstruction matching
    reviews_str = train['Reviews'].fillna('').astype(str)
    split_reviews = reviews_str.str.split(r'(?<=[.!?])\s+')

    train['tech'] = split_reviews.str[1].fillna('')
    train['body'] = split_reviews.str[1:].str.join(' ')

    # Build mapping dictionaries
    tech_to_course = train.groupby('tech')['Course'].first().to_dict()

    valid_bodies = train[train['body'] != '']
    body_to_indices = valid_bodies.groupby('body')['Index'].apply(list).to_dict()

    # Fit TF-IDF Vectorizer across Course Title + Review text
    vec = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    combined_corpus = train['Course'].fillna('') + " " + reviews_str
    train_mat = vec.fit_transform(combined_corpus)

    # Package objects into a lightweight binary payload
    index_data = {
        'train': train,
        'tech_to_course': tech_to_course,
        'body_to_indices': body_to_indices,
        'vec': vec,
        'train_mat': train_mat
    }

    # Save to pickle file
    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)

    print(f"✅ DONE! Successfully created '{output_path}'.")

if __name__ == "__main__":
    build_and_save_index()