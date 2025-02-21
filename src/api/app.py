from flask import Flask, request, jsonify
import torch
import base64
import io
from PIL import Image
import torchvision.transforms as transforms
from ..models.model import MNISTModel
from ..utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTModel().to(device)
model.load_state_dict(torch.load("models/trained_model.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get base64 encoded image from request
        image_data = request.json['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()
        
        logger.info(f"Predicted digit: {prediction}")
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
