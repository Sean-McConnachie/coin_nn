import os
import json

import torch
from torchvision import transforms
from torchvision.io import read_image

def load_details(details_path):
    with open(details_path, 'r') as f:
        details = json.load(f)
    return details

def get_transform(im_size, mean, std):
    transform = transforms.Compose([
        transforms.AutoAugment(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.Normalize(mean, std),
        transforms.RandomRotation(180),
    ])
    return transform

def load_model(base_model, weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('pytorch/vision:v0.10.0', base_model, pretrained=True)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(weights_path))
    return model

def load_image(image_path):
    image = read_image(image_path)
    return image

def image_to_tensor(image, transform):
    image = transform(image)
    image = image.unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    return image

def predict(model, image, classes):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return classes[str(predicted.item())]

if __name__ == "__main__":
    img_path = "coin_dataset/data/test/119/023__10 Cents_new_zealand.jpg"
    root = os.path.join("models", "v1")
    details = load_details(os.path.join(root, "details.json"))
    transform = get_transform(details['im_size'], details['mean'], details['std'])
    model = load_model(details['model'], os.path.join(root, details['weight_path']))
    image = image_to_tensor(load_image(img_path), transform)
    print(image.shape)
    prediction = predict(model, image, details['classes'])
    print(prediction)

