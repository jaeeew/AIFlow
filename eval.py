### eval.py (동일하게 사용) ###

import os
import numpy as np
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score

def load_mask(path):
    mask = Image.open(path).convert("L").resize((512, 512))
    return (np.array(mask) > 127).astype(np.uint8).flatten()

def evaluate(pred_dir, gt_dir):
    preds = sorted([f for f in os.listdir(pred_dir) if f.endswith("_pred.png")])
    iou_scores, dice_scores = [], []

    for fname in preds:
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname.replace("_pred.png", "_mask.png"))
        if not os.path.exists(gt_path):
            print(f"⚠️ 정답 없음: {gt_path}")
            continue
        pred_mask = load_mask(pred_path)
        gt_mask = load_mask(gt_path)
        iou = jaccard_score(gt_mask, pred_mask)
        dice = f1_score(gt_mask, pred_mask)
        iou_scores.append(iou)
        dice_scores.append(dice)
        print(f"{fname}: IoU={iou:.4f}, Dice={dice:.4f}")

    print("📊 전체 평균:")
    print(f"Avg IoU:  {np.mean(iou_scores):.4f}")
    print(f"Avg Dice: {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    evaluate("results", "data/vinylmasks_binary")
