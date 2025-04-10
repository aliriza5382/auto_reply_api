from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
import torch
import os

# NLTK stopwords indir
nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))

# Hafif transformer model (GPU istemez)
MODEL_NAME = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Firebase bağlantısı
cred = credentials.Certificate("businessmodel-ca2ce-firebase-adminsdk-fbsvc-54114476b3.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Flask app
app = Flask(__name__)

# Ön işleme fonksiyonu
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Embedding alma fonksiyonu (CLS token)
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

@app.route("/auto_reply", methods=["POST"])
def auto_reply():
    data = request.get_json()
    new_question = data.get("question")
    if not new_question:
        return jsonify({"error": "question field is missing"}), 400

    cleaned = preprocess(new_question)

    reply_docs = list(db.collection('supportReplies').stream())
    previous_questions = []
    previous_answers = []

    for reply_doc in reply_docs:
        reply_data = reply_doc.to_dict()
        rid = reply_data.get("requestId")
        answer = reply_data.get("reply")
        if not rid or not answer:
            continue

        request_doc = db.collection('supportRequests').document(rid).get()
        if not request_doc.exists:
            continue

        request_data = request_doc.to_dict()
        if request_data and 'request' in request_data:
            question_cleaned = preprocess(request_data['request'])
            previous_questions.append(question_cleaned)
            previous_answers.append(answer)

    if not previous_questions:
        return jsonify({"message": "no previous answers found"}), 200

    try:
        new_vec = get_embedding(cleaned)
        old_vecs = [get_embedding(q) for q in previous_questions]
        similarities = cosine_similarity([new_vec], old_vecs)[0]

        best_score = float(np.max(similarities))
        best_index = int(np.argmax(similarities))

        if best_score > 0.6:
            matched_response = previous_answers[best_index]
            return jsonify({
                "matched": True,
                "score": best_score,
                "reply": matched_response
            }), 200
        else:
            return jsonify({
                "matched": False,
                "score": best_score
            }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Render'da dışa açık port kullan
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
