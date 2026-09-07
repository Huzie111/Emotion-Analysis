"""
Multimodal Emotion Classification System
Deployed on Streamlit Cloud
Model: EfficientNet-B0 + BiLSTM (91.12% accuracy)
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gdown
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GOOGLE DRIVE FILE IDs - UPDATED WITH YOUR ACTUAL IDs
# ============================================================================

MODEL_FILE_ID = "11lYY2-0tXlF4mE1peB2ReQy9bMLlp2mp"
VOCAB_FILE_ID = "1r2mCVi-tVjeI18P2dBFFdlYAeHNuKnm-"

MODEL_FILE_NAME = "best_model_efficientnet_b0_bilstm.pt"
VOCAB_FILE_NAME = "vocabulary_3423.pth"

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden=128, dropout=0.7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, 2, bidirectional=True, 
                           batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden * 2
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context

def get_vision_encoder(name, pretrained=False):
    backbones = {
        'mobilenet_v2': (models.mobilenet_v2, 1280),
        'efficientnet_b0': (models.efficientnet_b0, 1280),
        'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, 1024),
    }
    
    if name not in backbones:
        raise ValueError(f"Unknown model: {name}")
    
    model_fn, dim = backbones[name]
    model = model_fn(pretrained=pretrained)
    
    if hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    elif hasattr(model, 'fc'):
        model.fc = nn.Identity()
    
    return model, dim

class MultimodalModel(nn.Module):
    def __init__(self, vision_name, text_encoder, dropout=0.7):
        super().__init__()
        
        self.vision, vdim = get_vision_encoder(vision_name, pretrained=False)
        self.vision_proj = nn.Sequential(
            nn.Linear(vdim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_encoder = text_encoder
        self.text_proj = nn.Sequential(
            nn.Linear(text_encoder.output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        fusion_dim = 512 + 256
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
    def forward(self, images, texts):
        v = self.vision(images)
        if v.dim() > 2:
            v = v.view(v.size(0), -1)
        v = self.vision_proj(v)
        
        t = self.text_encoder(texts)
        t = self.text_proj(t)
        
        fused = torch.cat([v, t], dim=1)
        return self.classifier(fused)

def create_text_encoder(text_type, vocab_size, hidden=128):
    if text_type == 'bilstm':
        return BiLSTMTextEncoder(vocab_size, hidden=hidden)
    else:
        raise ValueError(f"Unknown text encoder: {text_type}")

# ============================================================================
# LOAD FUNCTIONS WITH PROPER ERROR HANDLING
# ============================================================================

def download_file(file_id, file_name, description):
    """Download a file from Google Drive."""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)
        return True
    except Exception as e:
        st.error(f"❌ Failed to download {description}: {e}")
        return False

def load_vocabulary():
    """Load vocabulary with fallback."""
    try:
        # Check if file exists, if not download
        if not os.path.exists(VOCAB_FILE_NAME):
            with st.spinner(f"📥 Downloading vocabulary..."):
                success = download_file(VOCAB_FILE_ID, VOCAB_FILE_NAME, "vocabulary")
                if not success:
                    return None, None
        
        # Load vocabulary
        vocab_data = torch.load(VOCAB_FILE_NAME, map_location='cpu')
        
        # Extract word_to_idx
        if isinstance(vocab_data, dict):
            word_to_idx = vocab_data.get('word_to_idx', vocab_data)
        else:
            word_to_idx = vocab_data
        
        vocab_size = len(word_to_idx)
        return word_to_idx, vocab_size
        
    except Exception as e:
        st.error(f"❌ Error loading vocabulary: {e}")
        return None, None

def load_model(vocab_size):
    """Load model with proper error handling."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if file exists, if not download
        if not os.path.exists(MODEL_FILE_NAME):
            with st.spinner(f"📥 Downloading model..."):
                success = download_file(MODEL_FILE_ID, MODEL_FILE_NAME, "model")
                if not success:
                    return None, device
        
        # Create model
        text_enc = create_text_encoder('bilstm', vocab_size, hidden=128)
        model = MultimodalModel('efficientnet_b0', text_enc)
        
        # Load weights
        checkpoint = torch.load(MODEL_FILE_NAME, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, torch.device('cpu')

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    if isinstance(image, Image.Image):
        return image_transform(image).unsqueeze(0)
    return image

def tokenize_text(text, word_to_idx, max_len=100):
    """Tokenize text using vocabulary."""
    # Fallback if vocabulary is None
    if word_to_idx is None:
        tokens = text.lower().split()
        token_ids = [1] * len(tokens)  # All <UNK>
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        return torch.tensor(token_ids).unsqueeze(0)
    
    tokens = text.lower().split()
    token_ids = []
    for token in tokens:
        token_ids.append(word_to_idx.get(token, word_to_idx.get('<UNK>', 1)))
    
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        token_ids = token_ids + [0] * (max_len - len(token_ids))
    
    return torch.tensor(token_ids).unsqueeze(0)

def predict(model, image, text_tensor, device):
    with torch.no_grad():
        image = image.to(device)
        text_tensor = text_tensor.to(device)
        outputs = model(image, text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    return prediction, confidence, probabilities.cpu().numpy()

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Emotion Classifier - Children's Drawings",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2c3e50; text-align: center; }
    .sub-header { font-size: 1.1rem; color: #7f8c8d; text-align: center; margin-bottom: 2rem; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0; }
    .happy { background-color: #d4edda; border: 2px solid #28a745; }
    .sad { background-color: #f8d7da; border: 2px solid #dc3545; }
    .confidence-bar { height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin: 10px 0; }
    .confidence-fill { height: 100%; border-radius: 10px; transition: width 0.5s; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7rem; font-weight: bold; }
    .confidence-fill.happy { background: linear-gradient(90deg, #28a745, #20c997); }
    .confidence-fill.sad { background: linear-gradient(90deg, #dc3545, #e74c3c); }
    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #e9ecef; }
    .metric-value { font-size: 1.5rem; font-weight: 700; color: #2c3e50; }
    .metric-label { font-size: 0.8rem; color: #7f8c8d; }
    .stButton button { width: 100%; background: #3498db; color: white; font-weight: 600; padding: 10px; }
    .stButton button:hover { background: #2980b9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎨 Multimodal Emotion Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze children\'s drawings and self-reflections using EfficientNet-B0 + BiLSTM</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INITIALIZE RESOURCES
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Model Details")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Vision** | EfficientNet-B0 |
    | **Text** | BiLSTM |
    | **Accuracy** | 91.12% |
    | **Params** | 6.79M |
    | **Size** | 26.08 MB |
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("**KIDO Dataset**")
    st.markdown("- Children aged 6-14")
    st.markdown("- Uganda")
    st.markdown("- Happy / Sad classification")
    
    st.markdown("---")
    st.markdown("### ⚠️ Clinical Disclaimer")
    st.markdown("""
    This tool is for **screening purposes only** and is not a medical diagnostic device.
    **Rejection threshold:** Confidence < 85% → Manual review recommended.
    """)
    
    st.markdown("---")
    st.markdown("### 🔄 Loading Resources...")
    
    # Initialize variables
    word_to_idx = None
    vocab_size = 3423  # Your vocabulary size
    model = None
    device = torch.device('cpu')
    
    # Load vocabulary with error handling
    try:
        result = load_vocabulary()
        if result is not None and len(result) == 2:
            word_to_idx, vocab_size = result
            if word_to_idx is not None:
                st.success(f"✅ Vocabulary loaded! Size: {vocab_size}")
            else:
                st.warning("⚠️ Using fallback vocabulary")
                word_to_idx = {'<PAD>': 0, '<UNK>': 1}
                vocab_size = 3423
        else:
            st.warning("⚠️ Using fallback vocabulary")
            word_to_idx = {'<PAD>': 0, '<UNK>': 1}
            vocab_size = 3423
    except Exception as e:
        st.error(f"❌ Vocabulary error: {e}")
        st.warning("⚠️ Using fallback vocabulary")
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        vocab_size = 3423
    
    # Load model
    if word_to_idx is not None:
        try:
            model, device = load_model(vocab_size)
            if model is not None:
                st.success("✅ Model ready!")
            else:
                st.error("⚠️ Model not loaded")
        except Exception as e:
            st.error(f"❌ Model error: {e}")
    else:
        st.error("⚠️ Cannot load model without vocabulary")

# ============================================================================
# MAIN CONTENT
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Inputs")
    
    uploaded_image = st.file_uploader(
        "Upload a drawing (JPG/PNG)",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Uploaded Drawing", use_container_width=True)
    else:
        image = None
    
    st.markdown("---")
    text_input = st.text_area(
        "Enter self-reflection text",
        placeholder="e.g., I felt happy when I played with my friends today...",
        height=100
    )
    
    analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

with col2:
    st.markdown("### 📊 Results")
    
    if analyze_button and uploaded_image is not None and text_input.strip():
        if model is None:
            st.error("❌ Model not loaded. Please check the Google Drive connection.")
        else:
            with st.spinner("🧠 Analyzing emotion..."):
                try:
                    image_tensor = preprocess_image(image)
                    text_tensor = tokenize_text(text_input, word_to_idx, max_len=100)
                    prediction, confidence, probabilities = predict(model, image_tensor, text_tensor, device)
                    
                    class_names = ['Happy', 'Sad']
                    predicted_class = class_names[prediction]
                    confidence_percent = confidence * 100
                    
                    if predicted_class == 'Happy':
                        st.markdown(f"""
                        <div class="result-box happy">
                            <h1 style="font-size: 3rem;">😊</h1>
                            <h2>Happy</h2>
                            <p style="font-size: 1.2rem;">Confidence: {confidence_percent:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box sad">
                            <h1 style="font-size: 3rem;">😢</h1>
                            <h2>Sad</h2>
                            <p style="font-size: 1.2rem;">Confidence: {confidence_percent:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence bars
                    st.markdown("#### Confidence Distribution")
                    col1_bar, col2_bar = st.columns(2)
                    with col1_bar:
                        st.write("😊 Happy")
                        happy_pct = probabilities[0] * 100
                        fill_class = "happy" if predicted_class == 'Happy' else ""
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill {fill_class}" style="width: {happy_pct:.1f}%;">
                                {happy_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2_bar:
                        st.write("😢 Sad")
                        sad_pct = probabilities[1] * 100
                        fill_class = "sad" if predicted_class == 'Sad' else ""
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill {fill_class}" style="width: {sad_pct:.1f}%;">
                                {sad_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if confidence_percent < 85:
                        st.warning("⚠️ Low confidence (< 85%). Manual review recommended.")
                    else:
                        st.success("✅ High confidence prediction.")
                    
                    # Metrics
                    st.markdown("---")
                    st.markdown("#### 📈 Performance Metrics")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{confidence_percent:.1f}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{predicted_class}</div>
                            <div class="metric-label">Prediction</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">91.12%</div>
                            <div class="metric-label">Model Accuracy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")
    
    elif analyze_button:
        if uploaded_image is None:
            st.warning("⚠️ Please upload a drawing image.")
        if not text_input.strip():
            st.warning("⚠️ Please enter self-reflection text.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.8rem;">
    Multimodal Emotion Classification System &bull; EfficientNet-B0 + BiLSTM &bull; 91.12% Accuracy
</div>
""", unsafe_allow_html=True)
