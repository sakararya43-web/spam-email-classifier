from pathlib import Path
import pickle
import re

from flask import Flask, jsonify, request, render_template


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "email_spam_classifier.pkl"

app = Flask(__name__)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_for_pipeline(text):
    if isinstance(text, str):
        return preprocess_text(text)
    return [preprocess_text(item) for item in text]


class NotebookModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in {"preprocess_text", "preprocess_for_pipeline"}:
            return globals()[name]
        return super().find_class(module, name)


try:
    with MODEL_PATH.open("rb") as model_file:
        model = NotebookModelUnpickler(model_file).load()
except FileNotFoundError:
    model = None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": f"Model file not found: {MODEL_PATH}"}), 503

    data = request.get_json(silent=True) or {}
    email_text = data.get("text", data.get("email", ""))

    if not isinstance(email_text, str) or not email_text.strip():
        return jsonify({"error": "Please provide email text in the 'text' field."}), 400

    prediction = int(model.predict([email_text])[0])
    label = "spam" if prediction == 1 else "ham"

    return jsonify({"prediction": prediction, "label": label})


if __name__ == "__main__":
    app.run(debug=True)
