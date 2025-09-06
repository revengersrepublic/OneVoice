import os
import io
import torch
import pickle
import torchaudio
from flask import Flask, request, jsonify
from speechbrain.inference.speaker import SpeakerRecognition
import whisper

TEMPLATES_DIR = "speaker_templates"
UPLOADS_DIR = "temp_uploads"
LOCAL_MODEL_PATH = "pretrained_ecapa"
FINETUNED_MODEL_PATH = "speaker_model_noise_robust.pth"

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

print("[INFO] Loading base model structure...")

base_model = SpeakerRecognition.from_hparams(source=LOCAL_MODEL_PATH)

print(f"[INFO] Loading fine-tuned weights from: {FINETUNED_MODEL_PATH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
finetuned_state_dict = torch.load(FINETUNED_MODEL_PATH, map_location=torch.device(device))

base_model.mods.embedding_model.load_state_dict(finetuned_state_dict)

base_model.eval()
base_model.to(device)
print(f"[INFO] Fine-tuned model is ready for verification on device: {device}")

asr_model = whisper.load_model("tiny")

app = Flask(__name__)


def load_audio(file_obj):
    waveform, sr = torchaudio.load(file_obj)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform

@torch.no_grad()
def get_embedding(waveform):
    waveform = waveform.to(device)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)

    feats = base_model.mods.compute_features(waveform)
    embedding = base_model.mods.embedding_model(feats)
    embedding = embedding.squeeze()
    return embedding / embedding.norm(p=2)

def cosine_score(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1.view(-1), emb2.view(-1), dim=0).item()

def save_template(username, data):
    path = os.path.join(TEMPLATES_DIR, f"{username}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_template(username):
    path = os.path.join(TEMPLATES_DIR, f"{username}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@app.route("/enroll", methods=["POST"])
def enroll():
    username = request.form.get("username")
    password = request.form.get("password")
    files = request.files.getlist("audio")

    if not all([username, password, files]) or len(files) < 2:
        return jsonify({"error": "Request must include username, password, and at least 2 audio files."}), 400

    embeddings = []
    for f in files:
        wav = load_audio(f)
        emb = get_embedding(wav)
        embeddings.append(emb)

    avg_emb = torch.stack(embeddings).mean(dim=0)

    user_data = {
        "embedding": avg_emb.cpu(),
        "password": str(password).lower().replace(" ", "")
    }
    save_template(username, user_data)

    return jsonify({"message": f"Enrollment complete for user '{username}'"})

@app.route("/verify", methods=["POST"])
def verify():
    username = request.form.get("username")
    file = request.files.get("audio")

    if not username or not file:
        return jsonify({"error": "Request must include username and an audio file."}), 400

    user_data = load_template(username)
    if user_data is None:
        return jsonify({"error": f"User '{username}' not enrolled."}), 404

    template_embedding = user_data["embedding"].to(device)
    correct_password = user_data["password"]

    wav = load_audio(file)
    probe_emb = get_embedding(wav)

    score = cosine_score(probe_emb, template_embedding)
    speaker_decision = score >= 0.55

    tmp_path = os.path.join(UPLOADS_DIR, file.filename)
    file.seek(0)
    file.save(tmp_path)

    result = asr_model.transcribe(tmp_path)
    os.remove(tmp_path)

    transcript = result["text"].lower().replace(" ", "").replace(".", "").replace(",", "")
    keyword_decision = correct_password in transcript

    final_decision = "ACCEPT" if (speaker_decision and keyword_decision) else "REJECT"

    return jsonify({
        "decision": final_decision,
        "speaker_similarity_score": round(score, 4),
        "transcript": transcript,
        "speaker_match": speaker_decision,
        "keyword_match": keyword_decision,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
