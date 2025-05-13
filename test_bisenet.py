import os
import torch
from torchvision import transforms
from PIL import Image
from bisenetv2 import BiSeNetV2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiSeNetV2().to(device)
model.load_state_dict(torch.load("bisenet_vinyl.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

input_dir = "data/vinyldata"
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if not filename.endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, filename)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0, 0].cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask).save(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pred.png"))

    print(f"✅ 저장됨: {os.path.splitext(filename)[0]}_pred.png")
