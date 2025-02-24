from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
