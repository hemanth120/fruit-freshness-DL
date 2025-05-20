import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# --- Config ---
model_path = 'mobilenet_best.pth'
video_path = r'C:\Users\reddy\OneDrive\Desktop\fruit_freshness_detection\video3.mp4'  # or 0 for webcam
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = (224, 224)
frame_count=0
# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load class names from training dataset ---
class_names = sorted([
    'fresh_apples', 'fresh_bananas', 'fresh_cucumbers', 'fresh_oranges', 'fresh_tomatoes',
    'rotten_apples', 'rotten_bananas', 'rotten_cucumbers', 'rotten_oranges', 'rotten_tomatoes'
])

# --- Load model ---
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Video Capture ---
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 3 != 0:
        continue  # Skip this frame
    frame = cv2.resize(frame, (640, 480)) 
    # Optionally crop or resize if fruits are small; for now, assume multiple fruits fit in view
    h, w, _ = frame.shape
    step = 224  # sliding window size
    stride = 150  # how much to move window
    for y in range(0, h - step + 1, stride):
        for x in range(0, w - step + 1, stride):
            crop = frame[y:y + step, x:x + step]
            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)
                label = class_names[pred.item()]
                confidence = conf.item()

            if confidence > 0.85:
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + step, y + step), (0, 255, 0), 2)
                text = f'{label} ({confidence*100:.1f}%)'
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow('Fruit Freshness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
