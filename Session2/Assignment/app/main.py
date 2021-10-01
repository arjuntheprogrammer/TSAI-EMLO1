import sys
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from app.torch_utils import get_prediction, transform_image
from app.nn.constants import *
import uuid
import logging


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/media/<path:path>")
def static_dir(path):
    return send_from_directory("media", path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files.get('image_file', None)

    if uploaded_file is not None and uploaded_file.filename != "":
        if not allowed_file(uploaded_file.filename):
            return render_template(
                'index.html',
                error_msg="Format Not Supported!. Allowed: png, jpg, jpeg.")

        images_dir = "app/" + MEDIA_DIR
        os.makedirs(images_dir, exist_ok=True)

        try:
            file_name = (
                str(uuid.uuid4()) +
                "." + uploaded_file.filename.split(".")[-1])
            uploaded_file.save(
                images_dir + file_name)
        except:
            return render_template(
                'index.html',
                error_msg="file saving error!")

        try:
            tensor = transform_image(images_dir + file_name)
            print("image transform successful", flush=True)
        except:
            return render_template(
                'index.html',
                error_msg="error during tranformation!")

        try:
            prediction = get_prediction(tensor)
            print("image prediction successful", flush=True)

            return render_template('index.html', url=MEDIA_DIR + file_name, predicted_value=str(prediction.item()))

        except:
            return render_template(
                'index.html',
                error_msg="error during prediction!")

    else:
        return render_template(
            'index.html',
            error_msg="File Not Uploaded!")


if __name__ == '__main__':
    app.run(debug=True)

if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)
