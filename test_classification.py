# test_classification.py
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from model_classification import get_model
from train_classification import CustomDataset, test_df  # train_classification.py에서 정의된 클래스와 변수 불러오기

# label 매핑
label_map = {0: "plastic", 1: "paper"}

# 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
model.load_state_dict(torch.load("best_classification_model.pth", map_location=device))
model.eval()

# 테스트용 transform
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_loader = DataLoader(CustomDataset(test_df, eval_transform), batch_size=8, shuffle=False)

# 예측 및 시각화
images_to_plot = []
true_labels = []
pred_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).long().squeeze(1)

        for i in range(imgs.size(0)):
            img = imgs[i].cpu().permute(1, 2, 0).numpy()
            img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img = img.clip(0, 1)
            images_to_plot.append(img)
            true_labels.append(int(labels[i].item()))
            pred_labels.append(int(preds[i].item()))

plt.figure(figsize=(15, 8))
for i in range(len(images_to_plot)):
    plt.subplot(5, 6, i+1)
    plt.imshow(images_to_plot[i])
    true = label_map[true_labels[i]]
    pred = label_map[pred_labels[i]]
    color = 'green' if true == pred else 'red'
    plt.title(f"True: {true}\nPred: {pred}", color=color)
    plt.axis('off')

plt.suptitle("Test Set Prediction Results", fontsize=16)
plt.tight_layout()
plt.savefig("test_results_clf")
#plt.show()