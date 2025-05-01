from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('news_classification_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    author = request.form['author']
    title = request.form['title']
    combined = author + " " + title
    processed = preprocess(combined)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]

    result = "Fake News" if prediction == 1 else "Real News"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
