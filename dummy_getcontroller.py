from flask import Flask, request, jsonify
from fasttext_words_classifier import get_predict

app = Flask(__name__)


@app.route('/products', methods=['GET'])
def get_products():
    product_names = request.args.getlist('name')
    response = get_predict(product_names)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)