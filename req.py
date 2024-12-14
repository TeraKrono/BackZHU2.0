import numpy as np
import logging
from flask import Flask, request, jsonify
import io
import base64
from PIL import Image
import requests
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

client = chromadb.HttpClient(host="host.docker.internal", port=8800)

collection_name = "employee_collection"
collections = client.list_collections()
print(collections)
if collection_name not in [col.name for col in collections]:
    client.create_collection(name=collection_name)
else:
    print(f"Collection '{collection_name}' already exists.")

collection = client.get_collection(name="employee_collection")

MODEL_URL = "http://host.docker.internal:5000/compute_embedding"

help(client._query)

def get_embedding(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    data = {
        'image': image_base64
    }

    try:
        response = requests.post(MODEL_URL, json=data)
        if response.status_code == 200:
            return np.array(response.json()['embedding'])
        else:
            print(f"Error: {response.json()['error']}")
            return None
    except Exception as e:
        print(f"Error sending image to model: {e}")
        return None


def find_best_match(input_embedding, collection_name):
    collections = client.list_collections()
    collection_id = None

    for collection in collections:
        if collection.name == collection_name:
            collection_id = collection.id
            break

    if not collection_id:
        raise ValueError(f"Collection '{collection_name}' not found!")

    results = client._query(
        collection_id=collection_id,
        query_embeddings=[input_embedding],
        n_results=1
    )

    if not results['metadatas'] or len(results['metadatas']) == 0:
        return None

    metadata_list = results['metadatas'][0]
    distance = results['distances'][0]

    if isinstance(metadata_list, list) and len(metadata_list) > 0:
        metadata = metadata_list[0]  # Доступ до першого елемента списку
    else:
        metadata = {}

    return {
        "employee_id": metadata.get('employee_id', 'unknown'),
        "name": metadata.get('name', 'unknown'),
        "position": metadata.get('position', 'unknown'),
        "distance": distance
    }



@app.route("/verify", methods=["POST"])
def verify_employee():
    photo = request.files['photo']

    try:
        image = Image.open(photo.stream)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    input_embedding = get_embedding(image)
    if input_embedding is None:
        return jsonify({"error": "Failed to generate embedding for the image"}), 400

    best_match = find_best_match(input_embedding, "employee_collection")

    if best_match and best_match['distance'][0] <= 1.15:
        return jsonify({
            "employee_id": best_match['employee_id'],
            "name": best_match['name'],
            "position": best_match['position'],
            "distance": best_match['distance'][0]
        }), 200
    else:
        return jsonify({"error": "No matching employee found or distance too high"}), 405




@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo provided"}), 400

    photo = request.files['photo']
    try:
        image = Image.open(photo.stream)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

    data = {
        'image': image_base64
    }

    try:
        logging.info("Sending request to model...")
        response = requests.post(MODEL_URL, json=data)
        response.raise_for_status()
        model_result = response.json()

        if 'embedding' in model_result:
            embedding = model_result['embedding']

            employee_data = {
                'employee_id': request.form.get('employee_id', 'N/A'),
                'name': request.form.get('name', 'N/A'),
                'position': request.form.get('position', 'N/A')
            }

            collection.add(
                ids=[employee_data['employee_id']],
                embeddings=[embedding],
                metadatas=[employee_data]
            )

            logging.info(f"Employee data saved to ChromaDB: {employee_data['name']}")

            return jsonify({
                "message": "Photo uploaded successfully and processed by model",
                "model_result": model_result
            }), 200

        else:
            logging.error("Embedding not found in model response.")
            return jsonify({"error": "Embedding not found in model response"}), 400

    except requests.exceptions.RequestException as e:
        logging.error(f"Error contacting model: {str(e)}")
        return jsonify({"error": f"Failed to contact the model: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
