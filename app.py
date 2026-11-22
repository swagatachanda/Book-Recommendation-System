import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='Book Recommendation System', page_icon='📚', layout='centered')

@st.cache_resource
def load_data():
    data = pd.read_csv('dataset/Books.csv')
    data['clean_text'] = data['Book-Title'].fillna(' ')
    data = data.reset_index(drop=True)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    Xtfidf = tfidf.fit_transform(data['clean_text'])
    return data, Xtfidf

def recommend(item_name, data, X_tfidf, topn=5):
    match_items = data.index[data['Book-Title'] == item_name].tolist()
    if not match_items:
        return pd.DataFrame()
    index = match_items[0]
    similarity_score = cosine_similarity(X_tfidf[index], X_tfidf).flatten()
    similar_index = similarity_score.argsort()[::-1]
    similar_index = [i for i in similar_index if i !=index]
    top_index = similar_index[::topn]
    recommendation = data.iloc[top_index][['Book-Title']].copy()
    recommendation['Similarity Score'] = [similarity_score[i] for i in top_index]
    return recommendation.reset_index(drop=True)

data, X_tfidf = load_data()
st.title("📚 Book recommendation system")
st.write("Select a book below")
item_list = data['Book-Title'].dropna().unique().tolist()
selected_item = st.selectbox("Select a book", item_list)
if st.button("Generate Recommendation"):
    recommendations = recommend(selected_item, data, X_tfidf, topn=5)
    st.subheader("Recommend books")
    if recommendations.empty:
        st.warning("No recommendations found")
    else:
        for index, row in recommendations.iterrows():
            st.write(f"{index+1}. {row['Book-Title']}")
            st.caption(f"Similarity score: {row['Similarity Score']:.4f}")
