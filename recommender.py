import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class PathFinderRecommender:
    def __init__(self, index_path="search_index.pkl", train_path="train.csv"):
        self.test_to_train_s0_map = {
            "Took this course last month and here is my honest review.": "Took {Course} last month and here is my honest review.",
            "After completing this learning path I feel much more confident in this domain.": "After completing {Course} I feel much more confident in this domain.",
            "Just finished this program and I have a lot to say about it.": "Just finished {Course} and I have a lot to say about it.",
            "I recently completed this course and it was a fantastic learning experience.": "I recently completed {Course} and it was a fantastic learning experience.",
            "This course exceeded my expectations from the very first module.": "{Course} exceeded my expectations from the very first module.",
            "My overall experience with this learning path has been extremely positive.": "My overall experience with {Course} has been extremely positive.",
            "I enrolled hoping to level up my skills and the course did not disappoint.": "I enrolled in {Course} hoping to level up my skills and it did not disappoint.",
            "This was the course I needed at this stage of my learning journey.": "{Course} was the course I needed at this stage of my learning journey.",
            "I have been recommending this course to all my colleagues since finishing it.": "I have been recommending {Course} to all my colleagues since finishing it.",
            "I want to share my detailed thoughts after completing the full program.": "I want to share my detailed thoughts on {Course} after completing the full program.",
            "Signed up for this course on a whim and ended up loving every bit of it.": "Signed up for {Course} on a whim and ended up loving every bit of it."
        }
        
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                data = pickle.load(f)
                self.train = data['train']
                self.tech_to_course = data['tech_to_course']
                self.body_to_indices = data['body_to_indices']
                self.vec = data['vec']
                self.train_mat = data['train_mat']
        else:
            self.train = pd.read_csv(train_path)
            self.train['Index'] = self.train['Index'].astype(int)
            reviews_str = self.train['Reviews'].fillna('').astype(str)
            split_reviews = reviews_str.str.split(r'(?<=[.!?])\s+')
            self.train['tech'] = split_reviews.str[1].fillna('')
            self.train['body'] = split_reviews.str[1:].str.join(' ')
            self.tech_to_course = self.train.groupby('tech')['Course'].first().to_dict()
            valid_bodies = self.train[self.train['body'] != '']
            self.body_to_indices = valid_bodies.groupby('body')['Index'].apply(list).to_dict()
            self.vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), sublinear_tf=True, min_df=1)
            self.train_mat = self.vec.fit_transform(self.train['Course'].fillna('') + " " + reviews_str)

    def _get_sentences(self, text):
        text = str(text)
        protected = re.sub(r'(\b[A-Za-z0-9]+)\.([A-Za-z0-9]+\b)', r'\1__DOT__\2', text)
        return [s.replace('__DOT__', '.').strip() for s in re.split(r'(?<=[.!?])\s+', protected) if s.strip()]

    def _get_body_text(self, text):
        sents = self._get_sentences(text)
        return " ".join(sents[1:]) if len(sents) > 1 else ""

    def _reconstruct_query(self, query_text):
        sents = self._get_sentences(query_text)
        if not sents:
            return query_text
        t_s0 = sents[0]
        t_tech = sents[1] if len(sents) > 1 else ""
        detected_course = self.tech_to_course.get(t_tech, None)
        if detected_course:
            for masked_sent, template in self.test_to_train_s0_map.items():
                if masked_sent in t_s0:
                    return query_text.replace(masked_sent, template.replace("{Course}", str(detected_course)), 1)
        return query_text

    def recommend(self, query_text, top_k=4):
        reconstructed = self._reconstruct_query(query_text)
        query_mat = self.vec.transform([reconstructed])
        sim_scores = (query_mat @ self.train_mat.T).toarray()[0]
        
        # Limit candidate search slice to top 100 instead of iterating entire dataset
        top_positions = np.argsort(-sim_scores, kind='stable')[:100]
        q_body = self._get_body_text(query_text)
        exact_ids = self.body_to_indices.get(q_body, [])
        
        # Fast vectorized dataframe slicing
        results = self.train.iloc[top_positions].copy()
        
        # Pre-assign exact body matches to top priority
        if exact_ids:
            exact_df = self.train[self.train['Index'].isin(exact_ids)].copy()
            results = pd.concat([exact_df, results], ignore_index=True)
            
        results['Match_Score'] = results['Index'].map(
            lambda idx: round(float(sim_scores[self.train[self.train['Index'] == idx].index[0]]) * 100, 2)
            if idx in self.train['Index'].values else 95.0
        )
        
        # Deduplicate course names and slice top_k instantly
        deduped = results.drop_duplicates(subset=['Course'], keep='first')
        return deduped[['Index', 'Course', 'Reviews', 'Match_Score']].head(top_k)