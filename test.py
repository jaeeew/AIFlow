import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from train import UNet  # U-Net 클래스 import

# 테스트 데이터셋 정의
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.images[idx]

# 예측된 마스크 저장 함수
def save_mask(tensor, path, threshold=0.2):  # 🔽 여기 threshold 조절
    mask = (tensor.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


# 추론 함수
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = TestDataset("data/vinyldata", transform)
    loader = DataLoader(dataset, batch_size=1)

    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_vinyl.pth", map_location=device))
    model.eval()

    # 🔧 결과 폴더 초기화
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results")

    # 추론 부분 내부에서 sigmoid 적용
    with torch.no_grad():
        for img, name in loader:
            img = img.to(device)
            pred = model(img)

            # 🔽 sigmoid 적용
            pred = torch.sigmoid(pred)

            base_name = os.path.splitext(name[0])[0]
            save_path = os.path.join("results", f"{base_name}_pred.png")
            save_mask(pred, save_path)
            print(f"✅ 저장됨: {save_path}")

if __name__ == "__main__":
    run_inference()
