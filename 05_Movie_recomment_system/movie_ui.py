import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get absolute path of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to model.pkl
model_path = os.path.join(BASE_DIR, "model.pkl")

# Load model
model = joblib.load(model_path)

# Build similarity matrix dynamically
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(model["tags"]).toarray()
similar = cosine_similarity(vectors)

st.title("Movie Recommendation System 📽️")

# Movie selection
movie = st.selectbox("Enter your Movie", model["title"].values)

# Recommendation function
def recommend(movie):
    index = model[model["title"] == movie].index[0]
    movie_list = sorted(list(enumerate(similar[index])), 
                        reverse=True, key=lambda x: x[1])[1:11]
    rec_movies = [model.iloc[i[0]].title for i in movie_list]
    return rec_movies

# Button
if st.button("Recommend"):
    recommendations = recommend(movie)
    st.subheader("Top 10 Recommendations 🎬")
    for idx, title in enumerate(recommendations, start=1):
        st.write(f"{idx}. {title}")
