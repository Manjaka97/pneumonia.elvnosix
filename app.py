from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename
from datetime import datetime
from prediction import detect
from PIL import Image
from base64 import b64encode
import os, shutil, io
from google_drive_downloader import GoogleDriveDownloader as gdd


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# Downloading the weights because they are too large for github
if not os.path.exists('/yolov4.weights'):
	print('Downloading weights...')
	gdd.download_file_from_google_drive(file_id='1IfGBvFA7uGt2y6cmjJFW9uviXTSbzY55',
		
										dest_path='/yolov4.weights')
	print('Weights downloaded!')

# Cleaning temp dir for images, creating it if it does not exist
temp = '/static/temp'
if os.path.isdir(temp):
	shutil.rmtree(temp)
if not os.path.isdir(temp):
	os.mkdir(temp)


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}


@app.route('/', methods=['GET'])
def index():
	if request.method == 'GET':
		return render_template('index.html')


@app.route('/test', methods=['GET','POST'])
def test():
	if request.method == 'GET':
		return render_template('test.html')

	if request.method == 'POST':
		if 'photo' not in request.files:
			print('no file uploaded')
			return 'NO FILE UPLOADED'

		photo = request.files['photo']
		if allowed_file(photo.filename):
			timestamp = str(datetime.now())[:19]
			timestamp = timestamp.replace(':', '_')
			filename = os.path.join(temp, timestamp + secure_filename(photo.filename))
			photo.save(filename)

			detect(filename) # Outputs the result under the same filename

			image_pil = Image.open(filename)
			output = io.BytesIO()
			image_pil.save(output, format="JPEG") # Converts the image to a PIL image

			output.seek(0)
			output = b64encode(output.getvalue()) # Encodes the image to display with html
			os.remove(filename) # Deleting the file since it is already encoded in memory
			return render_template('result.html', output=output.decode('ascii'))
		return 'FILE NAME NOT ACCEPTED'

@app.route('/pneumonia', methods=['GET'])
def pneumonia():
	if request.method == 'GET':
		return render_template('pneumonia.html')


@app.route('/about', methods=['GET'])
def about():
	if request.method == 'GET':
		return render_template('about.html')


# @app.route('/training', methods=['GET'])
# def training():
# 	if request.method == 'GET':
# 		return render_template('training.html')


if __name__ == "__main__":
  app.run()