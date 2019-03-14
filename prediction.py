import torch
from model import preprocess_image, get_model

def predict_disease(image):
	model = get_model()
	tensor = preprocess_image(image)
	output = model(tensor)
	result = torch.max(output, 1)
	print(result)

