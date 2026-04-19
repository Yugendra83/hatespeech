import streamlit as st
import torch
import torch.nn as nn
import numpy as np

st.set_page_config(page_title="Hate Speech Detection")

st.write("🚀 App started")

# =========================
# SAFE LASER LOADING (NO FREEZE)
# =========================
@st.cache_resource
def load_laser():
    from laserembeddings import Laser
    return Laser()

laser = None  # lazy loading


def get_embedding(text):
    global laser
    if laser is None:
        laser = load_laser()
    return laser.embed_sentences([text], lang='en')[0]


# =========================
# ADAPTER LAYER
# =========================
class AdapterLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(AdapterLayer, self).__init__()
        self.down_project = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


# =========================
# MODEL CLASS (EXACT MATCH)
# =========================
class HateSpeechClassifier(nn.Module):
    def __init__(self, laser_dim=1024, lang_feat_dim=10, hidden_dim=256, adapter_dim=128, dropout=0.3):
        super(HateSpeechClassifier, self).__init__()

        input_dim = laser_dim + lang_feat_dim

        self.adapter1 = AdapterLayer(input_dim, adapter_dim, dropout)
        self.adapter2 = AdapterLayer(input_dim, adapter_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.adapter1(x)
        x = self.adapter2(x)
        output = self.classifier(x)
        return output


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = HateSpeechClassifier()
    checkpoint = torch.load("bestmodel.pth", map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


model = load_model()


# =========================
# LANGUAGE FEATURES
# =========================
def get_lang_features():
    vec = np.random.rand(10)
    return vec / vec.sum()


# =========================
# PREDICTION
# =========================
def predict(text):
    emb = get_embedding(text)
    lang_feat = get_lang_features()

    combined = np.concatenate([emb, lang_feat])

    tensor = torch.FloatTensor(combined).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor).item()

    return output


# =========================
# UI
# =========================
st.title("🧠 Multilingual Hate Speech Detection")

text = st.text_area("✍️ Enter text:")

if st.button("🚀 Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        score = predict(text)

        if score > 0.5:
            st.error("⚠️ Hate Speech Detected")
        else:
            st.success("✅ Safe Text")

        st.write(f"Confidence: {score:.2f}")
        st.progress(int(score * 100))