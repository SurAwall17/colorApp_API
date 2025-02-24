from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

############### ALGORITMA K-MEANS CLUSTERING ####################
@app.route('/analyze', methods=['POST'])
def analyze_image():
    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    k = 5  # Jumlah cluster

    # Analisis warna menggunakan K-Means
    reshaped_image = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped_image)

    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    label_counts = Counter(labels)

    total_count = sum(label_counts.values())
    color_percentages = {rgb_to_hex(colors[i]): count / total_count for i, count in label_counts.items()}

    return jsonify(color_percentages)

def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))



############### ALGORITMA C-BF ####################
# Membaca dataset dari CSV
df = pd.read_csv('/home/wallnine/Documents/Development/deteksi_warna_API/DataSet_Warna.csv', sep=';')
df['deskripsi'] = df['Mood'] + ' ' + df['Tema'] + ' ' + df['Suasana'] + ' ' + df['Kontras'] + ' ' + df['Popularitas'] + ' ' + df['Gaya Desain']

# Fungsi untuk rekomendasi warna
def rekomendasi_warna(mood_input, tema_input, suasana_input, kontras_input, popularitas_input, gaya_desain_input):
    input_deskripsi = f"{mood_input} {tema_input} {suasana_input} {kontras_input} {popularitas_input} {gaya_desain_input}"
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['deskripsi'].tolist() + [input_deskripsi])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    similar_indices = cosine_similarities.argsort()[0][-5:][::-1]
    recommended_colors = df.iloc[similar_indices][['Nama Warna', 'Kode Warna']]
    return recommended_colors.to_dict(orient='records')

# Route untuk API
@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    data = request.get_json()
    rekomendasi = rekomendasi_warna(
        data['mood'], data['tema'], data['suasana'], data['kontras'], data['popularitas'], data['gaya_desain']
    )
    return jsonify(rekomendasi)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
