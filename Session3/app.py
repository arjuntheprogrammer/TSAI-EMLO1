import os
from datetime import datetime
from math import floor

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory)
from werkzeug.utils import secure_filename

from constants import *
from json_db import *
from models import MobileNet

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app = Flask(__name__)

# It will allow below 16MB contents only, you can change it
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'super secret key'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = MobileNet()

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/media/<path:path>")
def static_dir(path):
    return send_from_directory("media", path)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/infer', methods=['POST'])
def success():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect('/')

        files = request.files.getlist('files[]')
        output_list = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                inference, confidence = model.infer(filepath)
                # make a percentage with 2 decimal points
                confidence = floor(confidence * 10000) / 100

                output_list.append({
                    "filename": filename,
                    "confidence": confidence,
                    "inference": inference,
                    "file_url": filepath,
                    "timestamp": datetime.now().isoformat()
                })
        print("output_list=", output_list)
        db_insert(output_list)

        previous_output_list = db_get()
        return render_template(
            'inference.html',
            output_list=output_list,
            previous_output_list=previous_output_list)


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 80))
    app.run(host='0.0.0.0', port=port, debug=True)
