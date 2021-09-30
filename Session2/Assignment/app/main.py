from os import error
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from app.torch_utils import get_prediction, transform_image
from app.nn.constants import *
import uuid

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

        try:
            file_path = (
                MEDIA_DIR + str(uuid.uuid4()) +
                "." + uploaded_file.filename.split(".")[-1])
            uploaded_file.save(
                "app/" + file_path)

            tensor = transform_image("app/" + file_path)
            prediction = get_prediction(tensor)

            return render_template('index.html', url=file_path, predicted_value=str(prediction.item()))

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
