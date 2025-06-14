from flask import Flask, request, jsonify
from waitress import serve
from .model import ALPR
import os
import cv2
import numpy as np
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

alpr = ALPR()

@app.route('/')
def home():
    return 'Hello, Flask!'

@app.route('/alpr', methods=['POST'])
def recognize_text():
    """
    ALPR API
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: Image file to recognize license plate
    responses:
      200:
        description: List of recognized texts
        schema:
          type: object
          properties:
            texts:
              type: array
              items:
                type: string
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    results = alpr.predict(img)

    return jsonify({'texts': results})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8000)
