# clf_model.py
import torch
from torchvision import transforms
from PIL import Image
from model_classification import get_model

# 모델 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model()
model.load_state_dict(torch.load("best_classification_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_material(img: Image.Image) -> str:
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 0 if prob < 0.5 else 1  # 0: plastic, 1: paper
        return 'plastic' if pred == 0 else 'paper'