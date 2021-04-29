import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras import applications
# from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import Flask, request
from io import BytesIO
from urllib.request import urlopen
from annoy import AnnoyIndex
import json
from collections import defaultdict

def extract_features(img_url, model, preprocess_input):
    input_shape = (224, 224, 3)
    res = urlopen(img_url).read()
    img = Image.open(BytesIO(res)).resize((input_shape[0], input_shape[1]))
    img_array = np.asarray(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)

    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features

model = tf.keras.models.load_model("./densenet201-epoch1_2.h5")
vector_len = 7
ann_n = 3
distance_metric = "euclidean"
annoy_index = AnnoyIndex(vector_len, distance_metric)
annoy_index.load("cat-pictures.annoy")

app = Flask(__name__)

@app.route('/inference', methods=['GET'])
def inference():
    features = extract_features(request.args["url"], model, applications.densenet.preprocess_input)

    scores = defaultdict(int)
    for idx, id in enumerate(annoy_index.get_nns_by_vector(features, ann_n)):
        scores[id] += ann_n - idx

    return json.dumps(scores)

@app.route('/update', methods=['POST'])
def update():
    params = json.loads(request.get_data())
    if len(params) == 0:
        return "No parameter"

    try:
        global annoy_index

        new_annoy_index = AnnoyIndex(vector_len, distance_metric)
        for id, url in params["urls"]:
            features = extract_features(url, model, applications.densenet.preprocess_input)
            new_annoy_index.add_item(id, features)
        new_annoy_index.build(10)

        annoy_index.unload()
        new_annoy_index.save("cat-pictures.annoy")
        annoy_index = new_annoy_index
        return "Update complete"
    except:
        return "Annoy error"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8880, threaded=False)