from flask import Flask, request, render_template

app = Flask(__name__)

from prediction import predict_disease

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'GET':
		return render_template('index.html')

	if request.method == 'POST':
		if 'file' not in request.files:
			print('no file uploaded')
			return
		file = request.files['file']
		image = file.read()
		prediction = predict_disease(image)
		return render_template('result.html', prediction=prediction) 


if __name__ == "__main__":
  app.run()