from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from dataset_class import ContamDataset
from UNet import UNet
from torch.utils.data import Dataset
import torch
import torch.nn as nn  # torch.nn 모듈을 nn으로 불러오기
import torch.optim as optim
import matplotlib.pyplot as plt

# 이미지 크기 변경 및 패딩을 위한 transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 이미지 크기를 256x256으로 리사이즈
    transforms.ToTensor(),  # 텐서로 변환
])

# Dataset 정의 시 transforms를 적용
train_dataset = ContamDataset(
    "dataset_split/train/images", "dataset_split/train/masks", transform=transform
)
val_dataset = ContamDataset(
    "dataset_split/val/images", "dataset_split/val/masks", transform=transform
)

# DataLoader 정의
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 모델, 손실 함수, 최적화 함수 정의
model = UNet()  # GPU로 모델을 옮김
criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실 함수
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(10):
    model.train()  # 모델을 훈련 모드로 설정
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs, masks  # GPU로 데이터 이동

        outputs = model(imgs)  # 모델을 통해 예측값을 얻음
        loss = criterion(outputs, masks)  # 손실 계산

        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트
        total_loss += loss.item()  # 손실값 누적

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")


model.eval()
with torch.no_grad():
    for imgs, masks in val_loader:
        imgs = imgs
        outputs = model(imgs)
        preds = (outputs > 0.5).float()

        for i in range(len(imgs)):
            plt.subplot(1,3,1)
            plt.imshow(imgs[i].cpu().permute(1,2,0))
            plt.title("Image")

            plt.subplot(1,3,2)
            plt.imshow(masks[i].cpu().squeeze(), cmap='gray')
            plt.title("Ground Truth")

            plt.subplot(1,3,3)
            plt.imshow(preds[i].cpu().squeeze(), cmap='gray')
            plt.title("Prediction")
            plt.show()
        break