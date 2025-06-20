# model_classification.py
import torch.nn as nn
import timm

def get_model():
    model = timm.create_model('efficientnet_b0', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model