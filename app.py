"""
Multimodal Emotion Classification System
Deployed on Streamlit Cloud
Model: EfficientNet-B0 + BiLSTM (91.12% accuracy)
Model stored on Google Drive - Downloaded at runtime
Vocabulary loaded from saved vocabulary.pth
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GOOGLE DRIVE FILE IDs
# ============================================================================

# Replace with your Google Drive file IDs
# To get this: Right-click on your file in Google Drive -> Share -> Copy link
# Example: https://drive.google.com/file/d/1ABC123XYZ456789/view -> ID is 1ABC123XYZ456789

MODEL_FILE_ID = "11lYY2-0tXlF4mE1peB2ReQy9bMLlp2mp"      # <-- REPLACE WITH YOUR ACTUAL MODEL FILE ID https://drive.google.com/file/d/1r2mCVi-tVjeI18P2dBFFdlYAeHNuKnm-/view?usp=drive_link
VOCAB_FILE_ID = "1r2mCVi-tVjeI18P2dBFFdlYAeHNuKnm-"      # <-- REPLACE WITH YOUR ACTUAL VOCAB FILE ID https://drive.google.com/file/d/11lYY2-0tXlF4mE1peB2ReQy9bMLlp2mp/view?usp=drive_link

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

class GRUTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden=128, dropout=0.7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden, 2, bidirectional=True, 
                         batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden * 2
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        gru_out, _ = self.gru(embedded)
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        context = torch.sum(attn_weights * gru_out, dim=1)
        return context

def get_vision_encoder(name, pretrained=False):
    if name == 'squeezenet1_1':
        model = models.squeezenet1_1(pretrained=pretrained)
        class SqueezeNetEncoder(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.features = base_model.features
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.feature_dim = 512
            def forward(self, x):
                x = self.features(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return x
        return SqueezeNetEncoder(model), 512
    
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
    elif text_type == 'gru':
        return GRUTextEncoder(vocab_size, hidden=hidden)
    else:
        raise ValueError(f"Unknown text encoder: {text_type}")

# ============================================================================
# LOAD VOCABULARY FROM GOOGLE DRIVE
# ============================================================================

@st.cache_resource
def load_vocabulary():
    """Download vocabulary from Google Drive and load it."""
    
    if not os.path.exists(VOCAB_FILE_NAME):
        st.info("📥 Downloading vocabulary from Google Drive...")
        
        try:
            url = f"https://drive.google.com/uc?id={VOCAB_FILE_ID}"
            gdown.download(url, VOCAB_FILE_NAME, quiet=False)
            st.success("✅ Vocabulary downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download vocabulary: {e}")
            return None
    
    try:
        # Load vocabulary
        vocab_data = torch.load(VOCAB_FILE_NAME, map_location='cpu')
        
        # Extract vocabulary
        if isinstance(vocab_data, dict):
            if 'word_to_idx' in vocab_data:
                word_to_idx = vocab_data['word_to_idx']
            else:
                word_to_idx = vocab_data
        else:
            word_to_idx = vocab_data
        
        vocab_size = len(word_to_idx)
        st.info(f"✅ Vocabulary loaded! Size: {vocab_size}")
        
        return word_to_idx, vocab_size
        
    except Exception as e:
        st.error(f"❌ Error loading vocabulary: {e}")
        return None, None

# ============================================================================
# LOAD MODEL FROM GOOGLE DRIVE
# ============================================================================

@st.cache_resource
def load_model_from_drive(vocab_size):
    """Download model from Google Drive and load it."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(MODEL_FILE_NAME):
        st.info("📥 Downloading model from Google Drive (this may take a moment)...")
        
        try:
            url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
            gdown.download(url, MODEL_FILE_NAME, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            return None, device
    
    # Model configuration - BEST PERFORMING MODEL
    vision_name = "efficientnet_b0"
    text_type = "bilstm"
    hidden = 128
    
    # Create text encoder with correct vocabulary size
    text_enc = create_text_encoder(text_type, vocab_size, hidden)
    
    # Create model
    model = MultimodalModel(vision_name, text_enc)
    
    # Load state dict
    try:
        checkpoint = torch.load(MODEL_FILE_NAME, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, device

# ============================================================================
# TEXT TOKENIZATION WITH VOCABULARY
# ============================================================================

def tokenize_text(text, word_to_idx, max_len=100):
    """Tokenize text using the loaded vocabulary."""
    
    # Convert to lowercase and split
    tokens = text.lower().split()
    
    # Convert to token IDs using vocabulary
    token_ids = []
    for token in tokens:
        if token in word_to_idx:
            token_ids.append(word_to_idx[token])
        else:
            token_ids.append(word_to_idx['<UNK>'])  # Unknown token
    
    # Truncate or pad to max length
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    else:
        # Pad with <PAD> token (index 0)
        token_ids = token_ids + [0] * (max_len - len(token_ids))
    
    return torch.tensor(token_ids).unsqueeze(0)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

# Image transforms (aligned with EfficientNet-B0 input)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    """Preprocess image for EfficientNet-B0 input."""
    if isinstance(image, Image.Image):
        return image_transform(image).unsqueeze(0)
    return image

def predict(model, image, text_tensor, device):
    """Run prediction on model."""
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .happy {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .sad {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .confidence-bar {
        height: 20px;
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.7rem;
        font-weight: bold;
    }
    .confidence-fill.happy {
        background: linear-gradient(90deg, #28a745, #20c997);
    }
    .confidence-fill.sad {
        background: linear-gradient(90deg, #dc3545, #e74c3c);
    }
    .stButton button {
        width: 100%;
        background: #3498db;
        color: white;
        font-weight: 600;
        padding: 10px;
    }
    .stButton button:hover {
        background: #2980b9;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎨 Multimodal Emotion Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze children\'s drawings and self-reflections using EfficientNet-B0 + BiLSTM</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Load Model and Vocabulary
# ============================================================================

with st.sidebar:
    st.markdown("### ℹ️ Model Details")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Vision** | EfficientNet-B0 |
    | **Text** | BiLSTM |
    | **Accuracy** | 91.12% |
    | **F1-Score** | 91.11% |
    | **Vocab Size** | 3,424 |
    | **Params** | 6.79M |
    | **Size** | 26.08 MB |
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Dataset")
    st.markdown("**KIDO Dataset**")
    st.markdown("- Children aged 6-14")
    st.markdown("- Uganda")
    st.markdown("- 5,430 labeled samples")
    st.markdown("- Happy / Sad classification")
    
    st.markdown("---")
    st.markdown("### ⚠️ Clinical Disclaimer")
    st.markdown("""
    This tool is for **screening purposes only** and is not a medical diagnostic device. 
    All predictions should be reviewed by qualified professionals. 
    
    **Rejection threshold:** Confidence < 85% → Manual review recommended.
    """)
    
    # Load vocabulary and model
    st.markdown("---")
    st.markdown("### 🔄 Loading Resources...")
    
    # Load vocabulary
    with st.spinner("Loading vocabulary..."):
        word_to_idx, vocab_size = load_vocabulary()
    
    # Load model if vocabulary loaded
    if word_to_idx is not None:
        with st.spinner("Loading model..."):
            model, device = load_model_from_drive(vocab_size)
            if model is not None:
                st.success("✅ All resources ready!")
            else:
                st.error("⚠️ Model not loaded")
    else:
        model = None
        device = torch.device('cpu')
        st.error("⚠️ Vocabulary not loaded")

# ============================================================================
# MAIN CONTENT - Two Columns
# ============================================================================

col1, col2 = st.columns([1, 1])

# ============================================================================
# COLUMN 1: INPUT SECTION
# ============================================================================

with col1:
    st.markdown("### 📤 Upload Inputs")
    
    # Image upload
    uploaded_image = st.file_uploader(
        "Upload a drawing (JPG/PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a child's drawing for emotion analysis"
    )
    
    # Display uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Uploaded Drawing", use_container_width=True)
    else:
        image = None
    
    # Text input
    st.markdown("---")
    text_input = st.text_area(
        "Enter self-reflection text",
        placeholder="e.g., I felt happy when I played with my friends today...",
        height=100,
        help="The child's description of their drawing or how they feel"
    )
    
    # Analyze button
    analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    # Instructions
    if not uploaded_image and not text_input:
        st.info("👆 Please upload a drawing and enter self-reflection text, then click Analyze.")

# ============================================================================
# COLUMN 2: RESULTS SECTION
# ============================================================================

with col2:
    st.markdown("### 📊 Results")
    
    # Initialize session state
    if 'result_displayed' not in st.session_state:
        st.session_state.result_displayed = False
    
    if analyze_button and uploaded_image is not None and text_input.strip():
        
        if model is None:
            st.error("❌ Model not loaded. Please check the Google Drive connection.")
            
        else:
            with st.spinner("🧠 Analyzing emotion..."):
                
                # Preprocess image
                image_tensor = preprocess_image(image)
                
                # Tokenize text using vocabulary
                text_tensor = tokenize_text(text_input, word_to_idx, max_len=100)
                
                # Run prediction
                prediction, confidence, probabilities = predict(
                    model, image_tensor, text_tensor, device
                )
                
                # Map prediction to class
                class_names = ['Happy', 'Sad']
                predicted_class = class_names[prediction]
                confidence_percent = confidence * 100
                
                # Store results in session state
                st.session_state.result_displayed = True
                st.session_state.predicted_class = predicted_class
                st.session_state.confidence = confidence_percent
                st.session_state.probabilities = probabilities[0]
                
                # Display result
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
                
                # Confidence bar
                st.markdown("#### Confidence Distribution")
                col1_bar, col2_bar = st.columns(2)
                with col1_bar:
                    st.write("😊 Happy")
                    fill_class = "happy" if predicted_class == 'Happy' else ""
                    happy_pct = probabilities[0] * 100
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill {fill_class}" style="width: {happy_pct:.1f}%;">
                            {happy_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2_bar:
                    st.write("😢 Sad")
                    fill_class = "sad" if predicted_class == 'Sad' else ""
                    sad_pct = probabilities[1] * 100
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill {fill_class}" style="width: {sad_pct:.1f}%;">
                            {sad_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence threshold warning
                if confidence_percent < 85:
                    st.warning("""
                    ⚠️ **Low Confidence:** The model is uncertain about this prediction 
                    (confidence < 85%). Manual review is recommended.
                    """)
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
                
    elif analyze_button:
        if uploaded_image is None:
            st.warning("⚠️ Please upload a drawing image.")
        if not text_input.strip():
            st.warning("⚠️ Please enter self-reflection text.")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.8rem;">
    Multimodal Emotion Classification System &bull; EfficientNet-B0 + BiLSTM &bull; 91.12% Accuracy
</div>
""", unsafe_allow_html=True)
