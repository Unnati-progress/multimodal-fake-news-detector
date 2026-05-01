
# ================= CELL 0 =================
import os
import re
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import requests

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from gnews import GNews
import cv2
import torchvision.models as models
import torchvision.transforms as transforms


# ================= CELL 1 =================
import pytesseract
import os
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"D:\chennai\New folder\tessdata\ocr\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

print("Tesseract exists:", os.path.exists(pytesseract.pytesseract.tesseract_cmd))

try:
    print("Version:", pytesseract.get_tesseract_version())
except Exception as e:
    print("Tesseract version check failed:", e)


# ================= CELL 2 =================
# Model paths (for Streamlit run from Project root)
TEXT_MODEL_PATH = "models/lr_model.pkl"
TEXT_VECTORIZER_PATH = "models/vectorizer.pkl"
IMAGE_MODEL_PATH = "models/image_clip_model.pkl"
VIDEO_MODEL_PATH = "models/video_cnn_lr_model.pkl"


# ================= CELL 3 =================
text_model = joblib.load(TEXT_MODEL_PATH)
text_vectorizer = joblib.load(TEXT_VECTORIZER_PATH)

print("Text model loaded")
print("Text vectorizer loaded")

# ================= CELL 4 =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

img_clip_model = joblib.load(IMAGE_MODEL_PATH)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Image CLIP model loaded")

# ================= CELL 5 =================
# ==============================
# Video model: MobileNetV2 CNN feature extractor + RandomForest classifier
# ==============================

video_rf_model = joblib.load(VIDEO_MODEL_PATH)

video_cnn_model = models.mobilenet_v2(pretrained=True)
video_cnn_model = video_cnn_model.features
video_cnn_model.eval()
video_cnn_model.to(device)

video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Video model loaded")


def extract_cnn_video_features(video_path, frames_per_video=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    frame_indexes = np.linspace(0, total_frames - 1, frames_per_video).astype(int)
    video_features = []

    with torch.no_grad():
        for idx in frame_indexes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = video_transform(frame).unsqueeze(0).to(device)

                features = video_cnn_model(input_tensor)
                features = torch.mean(features, dim=[2, 3])
                features = features.cpu().numpy().flatten()

                video_features.append(features)

    cap.release()

    if len(video_features) == 0:
        return None

    return np.mean(video_features, axis=0)


def predict_video_news(video_path):
    features = extract_cnn_video_features(video_path)

    if features is None:
        return "Error", 0.0

    features = features.reshape(1, -1)
    pred = video_rf_model.predict(features)[0]
    prob = video_rf_model.predict_proba(features)[0]

    label = "Real" if pred == 1 else "Fake"
    confidence = float(max(prob))

    return label, confidence


# ================= CELL 6 =================
from sentence_transformers import SentenceTransformer, util

rank_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Rank model loaded")

# ================= CELL 7 =================
# Keep your GNews API key in environment variable if possible.
# In Anaconda Prompt: set GNEWS_API_KEY=your_key_here
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")


# ================= CELL 8 =================
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\d", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= CELL 9 =================
def predict_text_news(claim_text):
    cleaned = clean_text(claim_text)
    vec = text_vectorizer.transform([cleaned])

    pred = text_model.predict(vec)[0]
    prob = text_model.predict_proba(vec)[0]

    label = "Real" if pred == 1 else "Fake"
    confidence = float(max(prob))

    return label, confidence

# ================= CELL 10 =================
def extract_clip_image_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = clip_model.vision_model(pixel_values=pixel_values)
            image_features = outputs.pooler_output

        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features.cpu().numpy().flatten()

    except Exception as e:
        print("Image error:", e)
        return None

# ================= CELL 11 =================
def extract_clip_text_features(text):
    try:
        inputs = clip_processor(
            text=[str(text)],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = clip_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_features = outputs.pooler_output

        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features.cpu().numpy().flatten()

    except Exception as e:
        print("Text error:", e)
        return None

# ================= CELL 12 =================
def predict_image_news(image_path, claim_text):
    img_feat = extract_clip_image_features(image_path)
    txt_feat = extract_clip_text_features(claim_text)

    if img_feat is None or txt_feat is None:
        return "Error", 0.0

    X = np.hstack([txt_feat.reshape(1, -1), img_feat.reshape(1, -1)])

    pred = img_clip_model.predict(X)[0]
    prob = img_clip_model.predict_proba(X)[0]

    label = "Real" if pred == 1 else "Fake"
    confidence = float(max(prob))

    return label, confidence

# ================= CELL 13 =================
def search_wikipedia(query, limit=3):
    url = "https://en.wikipedia.org/w/api.php"
    headers = {"User-Agent": "FakeNewsVerifier/1.0"}

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item["title"],
                "description": item["snippet"],
                "source": "Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                "publishedAt": ""
            })

        return results

    except:
        return []

# ================= CELL 14 =================
def search_gnews(query, limit=5):
    try:
        google_news = GNews(max_results=limit)
        google_news.api_key = GNEWS_API_KEY

        news = google_news.get_news(query)

        results = []
        for item in news:
            results.append({
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "source": item.get("publisher", {}).get("title", ""),
                "url": item.get("url", ""),
                "publishedAt": item.get("published date", "")
            })

        return results

    except:
        return []

# ================= CELL 15 =================
def retrieve_live_evidence(claim_text):
    return search_wikipedia(claim_text) + search_gnews(claim_text)

# ================= CELL 16 =================
def rank_evidence_by_similarity(claim_text, articles):
    if not articles:
        return []

    claim_emb = rank_model.encode(claim_text, convert_to_tensor=True)

    ranked = []
    for article in articles:
        text = article["title"] + " " + article["description"]
        emb = rank_model.encode(text, convert_to_tensor=True)

        score = util.cos_sim(claim_emb, emb).item()

        article["score"] = score
        ranked.append(article)

    return sorted(ranked, key=lambda x: x["score"], reverse=True)

# ================= CELL 17 =================
def extract_text_from_image(image_path):
    try:
        import cv2

        image = cv2.imread(image_path)
        if image is None:
            return ""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        text = pytesseract.image_to_string(gray)
        return text.strip()

    except Exception as e:
        print("OCR Error:", e)
        return ""

# ================= CELL 18 =================
def build_combined_query(claim_text, image_path):
    ocr_text = extract_text_from_image(image_path)

    if len(ocr_text.strip()) > 10:
        combined_query = claim_text + " " + ocr_text
    else:
        combined_query = claim_text

    return combined_query, ocr_text

# ================= CELL 19 =================
def final_verify_multimodal(claim_text, image_path=None, video_path=None):

    # Thresholds same as Prediction notebook
    REAL_THRESHOLD = 0.43
    LOW_EVIDENCE_THRESHOLD = 0.30

    # ------------------------------
    # 1. OCR + query creation
    # ------------------------------
    if image_path is not None:
        combined_query, ocr_text = build_combined_query(claim_text, image_path)
    else:
        combined_query = claim_text
        ocr_text = ""

    # ------------------------------
    # 2. Text model
    # ------------------------------
    text_label, text_conf = predict_text_news(combined_query)

    # ------------------------------
    # 3. Image model, optional
    # ------------------------------
    if image_path is not None:
        image_label, image_conf = predict_image_news(image_path, claim_text)
    else:
        image_label, image_conf = "Not Provided", 0.0

    # ------------------------------
    # 4. Video model, optional
    # ------------------------------
    if video_path is not None:
        video_label, video_conf = predict_video_news(video_path)
    else:
        video_label, video_conf = "Not Provided", 0.0

    # ------------------------------
    # 5. Live evidence retrieval
    # ------------------------------
    articles = retrieve_live_evidence(combined_query)
    ranked = rank_evidence_by_similarity(combined_query, articles)

    evidence_score = ranked[0]["score"] if ranked else 0.0
    evidence_count = len(ranked)

    # ------------------------------
    # 6. Final decision rules
    # ------------------------------
    model_votes = []

    if text_label in ["Real", "Fake"]:
        model_votes.append(text_label)

    if image_label in ["Real", "Fake"]:
        model_votes.append(image_label)

    if video_label in ["Real", "Fake"]:
        model_votes.append(video_label)

    fake_votes = model_votes.count("Fake")
    real_votes = model_votes.count("Real")

    if evidence_count > 0 and evidence_score >= REAL_THRESHOLD:
        final = "Real"
        reason = "Live evidence supports the claim."

    elif evidence_count == 0 or evidence_score < LOW_EVIDENCE_THRESHOLD:
        if fake_votes >= 2:
            final = "Fake"
            reason = "Weak/no evidence and multiple models predict Fake."
        elif real_votes >= 2:
            final = "Suspicious"
            reason = "Models lean Real, but live evidence is weak."
        else:
            final = "Suspicious"
            reason = "Weak evidence and mixed model signals."

    else:
        if fake_votes > real_votes:
            final = "Suspicious"
            reason = "Partial evidence exists, but model signals lean Fake."
        else:
            final = "Suspicious"
            reason = "Partial evidence but not strong enough."

    return {
        "final": final,
        "final_decision": final,
        "reason": reason,
        "claim_text": claim_text,
        "ocr_text": ocr_text,
        "combined_query": combined_query,
        "text_label": text_label,
        "text_conf": text_conf,
        "image_label": image_label,
        "image_conf": image_conf,
        "video_label": video_label,
        "video_conf": video_conf,
        "evidence_score": evidence_score,
        "top_evidence": ranked[:3]
    }

# ================= CELL 20 =================
def show_result(result):
    print("FINAL:", result["final"])
    print("Reason:", result["reason"])
    print("Claim:", result["claim_text"])
    print("OCR Text:", result["ocr_text"])
    print("Combined Query:", result["combined_query"])
    print("Text Model:", result["text_label"], "| Confidence:", round(result["text_conf"], 3))
    print("Image Model:", result["image_label"], "| Confidence:", round(result["image_conf"], 3))
    print("Video Model:", result["video_label"], "| Confidence:", round(result["video_conf"], 3))
    print("Evidence Score:", result["evidence_score"])

    print("\nTop Evidence:\n")
    for e in result["top_evidence"]:
        print(e["title"])
        print(e["url"])
        print("-" * 50)
