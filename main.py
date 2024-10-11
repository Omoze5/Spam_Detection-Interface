import os
from flask import Flask, url_for, render_template, request
import pickle
import spacy

app = Flask(__name__)

# Load the model and Spacy language model
nlp = spacy.load("en_core_web_sm")
model = pickle.load(open('SVC_model.pkl', 'rb'))

# Preprocessing function
def preprocess(txt):
    doc = nlp(txt)
    filtered_token = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_token.append(token.lemma_.lower())
    return " ".join(filtered_token)

# Home route
@app.route('/')
def Home():
    # Render home page 
    return render_template('SPAM.html')

# Submit route for predictions
@app.route('/submit', methods=['POST'])
def submit():
    label_mapping = {0: "Ham", 1: "Spam"}  # Example mapping for prediction labels
    text = request.form["review"]  # Get the email text from the form
    preprocessed_text = preprocess(text)  # Preprocess the email text
    
    # Predict whether the email is spam or ham
    prediction = model.predict([preprocessed_text])
    predicted_label = label_mapping[prediction[0]]


    # Render the result back to the user, along with updated email lists
    return render_template("SPAM.html", Predicted=predicted_label,
                           submitted=True)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
