from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_similarity import Similary

app = Flask(__name__)


@app.route("/")
def hello():
    return "Welcome to machine learning model APIs!"

@app.route("/similarity", methods =['POST'])
def similarity():
    data = request.get_json()
    model = Similary(data['arr_for_matching'],data['arr_database'],data['arr_preprocess'],data['threshold'],data['NER'],data['Device'])
    matches = model.predict()
    return jsonify(matches)


if __name__ == '__main__':
    app.run(debug=False,threaded=True)