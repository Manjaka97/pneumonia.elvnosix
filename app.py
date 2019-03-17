from flask import Flask, request, render_template

app = Flask(__name__)

from prediction import predict_disease

@app.route('/', methods=['GET'])
def index():
	if request.method == 'GET':
		return render_template('index.html')

@app.route('/test', methods=['GET','POST'])
def test():
	if request.method == 'GET':
		return render_template('test.html')

	if request.method == 'POST':
		print(request.files)
		if 'file' not in request.files:
			print('no file uploaded')
			return 'NO FILE UPLOADED'
		file = request.files['file']
		image = file.read()
		prediction = predict_disease(image)
		return render_template('result.html', prediction=prediction)

@app.route('/pneumonia', methods=['GET'])
def pneumonia():
	if request.method == 'GET':
		return render_template('pneumonia.html')

@app.route('/about', methods=['GET'])
def about():
	if request.method == 'GET':
		return render_template('about.html')


if __name__ == "__main__":
  app.run()