from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import io
import json
import os

app = Flask(__name__)

model = deeplabv3_resnet101(pretrained=False)
num_classes = 1
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

model.load_state_dict(torch.load('segmentation.pth', map_location=torch.device('cpu')), strict=False)
model.eval()
model.to(torch.device('cpu'))

output_dir = '/home/app/data'
os.makedirs(output_dir, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)['out']
            predicted_mask = torch.sigmoid(output).numpy()

        filename = f"mask_{file.filename.split('.')[0]}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, 'w') as f:
            json.dump({'mask': predicted_mask.tolist()}, f)

        response = {'message': 'Mask generated successfully', 'file_path': file_path}
        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9001)
