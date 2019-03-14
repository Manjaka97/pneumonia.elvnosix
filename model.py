import io

import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

def get_model():
	weights = 'resnet18-2.pth'
	model = models.resnet18(pretrained=True)
	num_features = model.fc.in_features
	model.fc = nn.Linear(num_features, 2)
	model.load_state_dict(torch.load(weights, map_location='cpu'))
	model.eval()
	return model

def preprocess_image(image):
	image_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
								           transforms.Resize(256),
								           transforms.CenterCrop(224),
								           transforms.ToTensor(),
								           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    								  ])

	image_tensor = Image.open(io.BytesIO(image))
	return image_transforms(image_tensor).unsqueeze(0)