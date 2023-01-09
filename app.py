from flask import Flask, request, jsonify
import pickle
from model import  SentimentAnalysis

# Load the model from a file
with open('model.pkl', 'rb') as file:
    sentimentAnalysis = pickle.load(file)


app = Flask(__name__)

@app.route('/sentiment', methods = ['POST'])
def getSentiment():
    text = request.json['text']
    sentiment =  sentimentAnalysis.predict(text)
    print(sentiment)
    return jsonify({"response" : "your sentiment are "+ str(sentiment)})


if __name__ == "__main__":
    app.run(debug=True, port = 4000)