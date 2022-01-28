import os.path
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
from zipfile import ZipFile

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello():
    return "Hello"


@app.route('/api/v1/search_text/', methods=['POST'])
def search_text():
    if request.method == 'POST':
        indexer_name = request.json["indexer_name"]
        text_query = request.json["text_query"]
        top_k = request.json["top_k"]
        return jsonify({"links": {
            "link1": "https://images.pexels.com/photos/9040070/pexels-photo-9040070.jpeg",
            "link2": "https://images.pexels.com/photos/9207676/pexels-photo-9207676.jpeg",
            "link3": "https://images.pexels.com/photos/10368603/pexels-photo-10368603.jpeg",
            "link4": "https://images.pexels.com/photos/10849829/pexels-photo-10849829.jpeg",
            "link5": "https://images.pexels.com/photos/10050311/pexels-photo-10050311.jpeg"
        }})


@app.route('/api/v1/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        extension = os.path.splitext(filename)[1]
        if file and allowed_file(filename):
            if extension in ['png', 'jpg', 'jpeg', 'gif']:
                filename = secure_filename(file.filename)
                # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.save('\\temp', filename)
            elif extension == 'zip':
                filename = secure_filename(file.filename)
                with ZipFile(filename) as zf:
                    for file in zf.namelist():
                        filename = secure_filename(file.filename)
                        extension = os.path.splitext(filename)[1]
                        if extension in ['png', 'jpg', 'jpeg', 'gif']:
                            filename = secure_filename(file.filename)
                            file.save('\\temp', filename)
            return "File saved"



if __name__ == '__main__':
    app.run(debug=True)
