from flask import Flask, request, send_from_directory
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)

data_directory = '../data'  # Directory relative to the script where JSON files are stored

@app.route('/process_json_to_image', methods=['POST'])
def process_json_to_image():
    try:
        json_filename = request.args.get('filename')
        if not json_filename:
            return "Filename parameter is missing", 400

        json_path = os.path.join(data_directory, json_filename)
        
        if not os.path.exists(json_path):
            return f"No such file: {json_path}", 404

        with open(json_path, 'r') as file:
            data = json.load(file)
        
        mask = np.array(data['mask'])
        print("Mask shape:", mask.shape)

        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        elif mask.ndim == 4:
            mask = mask[0, :, :, 0]
        elif mask.ndim != 2:
            return "Unsupported number of dimensions: " + str(mask.ndim), 400
        
        image_data = (mask * 255).astype(np.uint8)
        image = Image.fromarray(image_data, 'L')
        
        image_filename = f"{os.path.splitext(json_filename)[0]}_mask.png"
        image.save(os.path.join(data_directory, image_filename))
        
        return send_from_directory(directory=data_directory, filename=image_filename)
    
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
