from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("E:/data.csv")

# Preprocess the data
selected_features = ['title', 'authors', 'categories', 'published_year']
for feature in selected_features:
    df[feature] = df[feature].fillna('')
combined_features = df['title'] + ' ' + df['categories'] + ' ' + df['authors'] + ' ' + f"{df['published_year']}"
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors, feature_vectors)

# Endpoint for recommending books
@app.route('/recommend', methods=['GET', 'POST'])
def recommend_books():
    if request.method == 'GET':
        book_name = request.args.get('book_name', '')
    elif request.method == 'POST':
        book_name = request.args.get('book_name', '')
    else:
        return jsonify({'error': 'Unsupported HTTP method.'}), 405

    if not book_name:
        return jsonify({'error': 'Please provide a book name.'}), 400
    
    list_of_all_titles = df['title'].tolist()
    find_close_match = difflib.get_close_matches(book_name, list_of_all_titles)
    if not find_close_match:
        return jsonify({'error': 'No close match found for the provided book name.'}), 404
    
    close_match = find_close_match[0]
    index_of_the_book = df[df.title == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_the_book]))
    sorted_similar_books = sorted(similarity_score, key=lambda x: x[1], reverse=True)[:29]
    
    recommended_books = []
    for book in sorted_similar_books:
        index = book[0]
        title_from_index = df[df.index == index]['title'].values[0]
        recommended_books.append({'title': title_from_index})
    
    return jsonify({'recommended_books': recommended_books}), 200



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
