import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F

# Configuration
UPLOAD_FOLDER = r'C:\Users\reddy\OneDrive\Desktop\fruit_freshness_detection\upload'
MODEL_PATH = 'mobilenet_best.pth'
CLASS_NAMES = ['fresh_apples', 'fresh_bananas',  'fresh_cucumber', 'fresh_oranges','fresh_tomatoes', 
               'rotten_apples', 'rotten_bananas', 'rotten_cucumber','rotten_oranges','rotten_tomatoes'  ]
               
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.mobilenet_v2(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = CLASS_NAMES[pred.item()]
        confidence = confidence.item() * 100
        fruit, status = label.split("_")
        status = status.capitalize()
        fruit = fruit.capitalize()
        return f"{fruit} – {status}", confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            file.save(path)
            prediction, confidence = predict_image(path)
    return render_template('index.html', prediction=prediction, confidence=confidence, image_filename=image_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
