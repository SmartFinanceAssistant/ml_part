from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/products', methods=['GET'])
def get_products():
    product_names = request.args.getlist('name')
    response = {
        'product_types': product_names
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)