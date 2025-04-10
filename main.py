from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer, util
import string, nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('turkish'))
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

cred = credentials.Certificate("businessmodel-ca2ce-firebase-adminsdk-fbsvc-54114476b3.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

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

    all_sentences = [cleaned] + previous_questions
    embeddings = model.encode(all_sentences, convert_to_tensor=True)
    similarities = util.cos_sim(embeddings[0], embeddings[1:])[0]

    best_score = float(similarities.max())
    best_index = int(similarities.argmax())

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Render özel port ister!
