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
		if 'file' not in request.files:
			print('no file uploaded')
			return
		file = request.files['file']
		image = file.read()
		prediction = predict_disease(image)
		return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
  app.run()