import torch
from model import preprocess_image, get_model

def predict_disease(image):
	model = get_model()
	tensor = preprocess_image(image)
	output = model(tensor)
	_, prediction = torch.max(output, 1)
	diagnostic = ''
	if prediction.item() == 0:
		diagnostic = 'No sign of Pneumonia Detected'
	if prediction.item() == 1:
		diagnostic = 'Sign of Pneumonia Detected'
	return prediction

