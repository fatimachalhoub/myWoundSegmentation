from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

@app.route('/convert_mask', methods=['POST'])
def convert_mask():
    try:
        # Assume the mask is sent as a base64 encoded image
        data = request.json['mask']
        mask_data = base64.b64decode(data)
        mask_image = Image.open(io.BytesIO(mask_data))
        
        # Here you can process the mask image, e.g., apply a color map
        processed_image = apply_color_map(mask_image)

        # Save or send image directly
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        response = base64.b64encode(img_byte_arr).decode('ascii')
        return jsonify({'image': response})

    except Exception as e:
        return jsonify({'error': str(e)})

def apply_color_map(image):
    # Convert image to numpy array
    image_array = np.array(image)
    # Apply a color map - for example, converting grayscale to pseudo-color
    colored_image = Image.fromarray(np.uint8(plt.cm.jet(image_array)*255))
    return colored_image

if __name__ == '__main__':
    app.run(debug=True, port=5001)

