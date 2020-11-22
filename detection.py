# Import libraries
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from imageai.Detection import ObjectDetection


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# UPLOAD_FOLDER = os.path.join(APP_ROOT, 'F:/Leaf-Detection-master/templates/tmp/')
# UPLOAD_FOLDER = os.path.join(APP_ROOT, 'F:/Leaf-Detection-master/static/')
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['jpg'])



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/upload_image", methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect('/')
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect('/')
        if file and allowed_file(file.filename):
            if request.form['submit_button'] == 'Detect':
                file.filename = "image.jpg"
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            execution_path = os.getcwd()
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
            detector.loadModel()
            detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,
                                                                                  UPLOAD_FOLDER + "image.jpg"),
                                                         output_image_path=os.path.join(execution_path,
                                                                                        UPLOAD_FOLDER + "imagenew.jpg")
                                                         )
            image_result = []
            for eachObject in detections:
                key = eachObject["name"]
                value = eachObject["percentage_probability"]

                result = {key: value}
                print(result)
                image_result.append(result)
            return render_template("result.html", image_result=image_result)

        else:
            flash('Allowed file types are .jpg')
            return redirect('/')


if __name__ == '__main__':
    app.secret_key = 'SECRET KEY'
    app.run(port=5000, debug=True)
