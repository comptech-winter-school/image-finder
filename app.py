from flask import Flask, jsonify, request

app = Flask(__name__)


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


if __name__ == '__main__':
    app.run(debug=True)
