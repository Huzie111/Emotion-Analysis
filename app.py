"""
L32 Layer-Wise Grad-CAM Streamlit App
Model: MobileNetV2 + BiLSTM (92.69% Validation Accuracy)
Layer 32 (L32) captures high-level concepts and emotion-relevant regions
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Emotion Analysis - L32 Grad-CAM",
    page_icon="🎨",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS - NO SIDEBAR
# ============================================================================

st.markdown("""
<style>
    /* Hide sidebar */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Main content full width */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px 0;
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
        height: 16px;
        background: #e9ecef;
        border-radius: 8px;
        overflow: hidden;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.6rem;
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
        font-size: 1rem;
    }
    .stButton button:hover {
        background: #2980b9;
    }
    .model-info {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
        font-size: 0.8rem;
    }
    .model-info table {
        width: 100%;
        font-size: 0.8rem;
    }
    .model-info td {
        padding: 2px 6px;
    }
    .model-info .label {
        font-weight: 600;
        color: #495057;
    }
    .word-highlight {
        display: inline-block;
        padding: 2px 6px;
        margin: 1px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .word-highlight.high {
        background: #dc3545;
        color: white;
    }
    .word-highlight.medium {
        background: #ffc107;
        color: #333;
    }
    .word-highlight.low {
        background: #e9ecef;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🎨 Emotion Analysis from Children\'s Drawings</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MobileNetV2 + BiLSTM | L32 Layer-Wise Grad-CAM Explainability</div>', unsafe_allow_html=True)

# ============================================================================
# MODEL COMPONENTS
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
# L32 GRAD-CAM
# ============================================================================

def get_target_layer_l32(model):
    """Get the 32nd convolutional layer (L32) from the vision encoder."""
    
    conv_layers = []
    
    def collect_layers(module):
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                conv_layers.append(child)
            collect_layers(child)
    
    collect_layers(model.vision)
    
    # L32 = 32nd convolutional layer (0-indexed)
    if len(conv_layers) > 32:
        st.info(f"Found {len(conv_layers)} Conv2d layers. Using Layer 32 (L32)")
        return conv_layers[32]
    else:
        st.warning(f"Only {len(conv_layers)} Conv2d layers found. Using last layer as fallback.")
        return conv_layers[-1]

class L32GradCAM:
    """L32 Layer-Wise Grad-CAM implementation."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.target_layer = get_target_layer_l32(model)
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_heatmap(self, image, text_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        image = image.to(self.device)
        text_tensor = text_tensor.to(self.device)
        image.requires_grad = True
        
        output = self.model(image, text_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return None, target_class
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        heatmap = cam.squeeze().detach().cpu().numpy()
        return heatmap, target_class

def resize_heatmap(heatmap, target_size):
    """Resize heatmap using PyTorch interpolate."""
    heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    return resized.squeeze().cpu().numpy()

def create_overlay(image, heatmap, alpha=0.6):
    """Create heatmap overlay."""
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        img = image.squeeze().detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
    else:
        img = np.array(image) / 255.0
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
    
    # Resize heatmap
    target_size = (img.shape[0], img.shape[1])
    heatmap_resized = resize_heatmap(heatmap, target_size)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    
    # Create RGB heatmap
    from matplotlib import cm
    heatmap_rgb = cm.jet(heatmap_resized)[:, :, :3]
    
    # Overlay
    overlay = (1 - alpha) * img + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, heatmap_resized

def generate_l32_gradcam(model, image, text_tensor, device, target_class=None):
    """Generate L32 Grad-CAM explanation."""
    try:
        gradcam = L32GradCAM(model, device)
        heatmap, target_class = gradcam.generate_heatmap(image, text_tensor, target_class)
        
        if heatmap is None:
            return None, None
        
        overlay, heatmap_resized = create_overlay(image, heatmap, alpha=0.6)
        return overlay, heatmap_resized
        
    except Exception as e:
        st.warning(f"L32 Grad-CAM error: {e}")
        return None, None

# ============================================================================
# LIME FOR TEXT EXPLANATION
# ============================================================================

def generate_lime_text_explanation(text, model, word_to_idx, device, max_len=50):
    words = text.lower().split()
    if len(words) == 0:
        return None, []
    
    def preprocess_text(text, word_to_idx, max_len=50):
        if word_to_idx is None:
            tokens = text.lower().split()
            token_ids = [1] * len(tokens)
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            else:
                token_ids = token_ids + [0] * (max_len - len(token_ids))
            return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        
        tokens = text.lower().split()
        token_ids = []
        for token in tokens:
            token_ids.append(word_to_idx.get(token, word_to_idx.get('<UNK>', 1)))
        
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        else:
            token_ids = token_ids + [0] * (max_len - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    
    text_tensor = preprocess_text(text, word_to_idx, max_len)
    text_tensor = text_tensor.to(device)
    dummy_image = torch.zeros(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_image, text_tensor)
        base_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        base_pred = np.argmax(base_probs)
    
    word_importance = []
    for i, word in enumerate(words):
        perturbed_words = words[:i] + words[i+1:]
        perturbed_text = ' '.join(perturbed_words)
        
        if len(perturbed_text.strip()) == 0:
            continue
        
        perturbed_tensor = preprocess_text(perturbed_text, word_to_idx, max_len)
        perturbed_tensor = perturbed_tensor.to(device)
        
        with torch.no_grad():
            perturbed_outputs = model(dummy_image, perturbed_tensor)
            perturbed_probs = torch.softmax(perturbed_outputs, dim=1).cpu().numpy()[0]
        
        importance = abs(base_probs[base_pred] - perturbed_probs[base_pred])
        word_importance.append((word, importance))
    
    if len(word_importance) > 0:
        max_imp = max([imp for _, imp in word_importance])
        if max_imp > 0:
            word_importance = [(w, imp / max_imp) for w, imp in word_importance]
    
    return base_pred, word_importance

# ============================================================================
# LOAD MODEL AND VOCABULARY
# ============================================================================

@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabulary from Google Drive."""
    
    import gdown
    import requests
    
    # File IDs
    MODEL_FILE_ID = "11lYY2-0tXlF4mE1peB2ReQy9bMLlp2mp"
    VOCAB_FILE_ID = "1r2mCVi-tVjeI18P2dBFFdlYAeHNuKnm-"
    
    MODEL_FILE_NAME = "MM_MobileNetV2_BiLSTM_final.pt"
    VOCAB_FILE_NAME = "vocabulary.pth"
    
    def download_file(file_id, file_name, description):
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, file_name, quiet=False)
            if os.path.exists(file_name):
                return True
            return False
        except:
            try:
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                session = requests.Session()
                response = session.get(url, stream=True)
                if 'confirm' in response.text:
                    confirm_match = re.search(r'confirm=([^&]+)', response.text)
                    if confirm_match:
                        confirm_token = confirm_match.group(1)
                        url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                        response = session.get(url, stream=True)
                if response.status_code == 200:
                    with open(file_name, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return True
                return False
            except:
                return False
    
    # Download vocabulary
    if not os.path.exists(VOCAB_FILE_NAME):
        st.info("📥 Downloading vocabulary...")
        success = download_file(VOCAB_FILE_ID, VOCAB_FILE_NAME, "vocabulary")
        if not success:
            st.error("Failed to download vocabulary")
            return None, None, None
    
    # Download model
    if not os.path.exists(MODEL_FILE_NAME):
        st.info("📥 Downloading model (this may take a few minutes)...")
        success = download_file(MODEL_FILE_ID, MODEL_FILE_NAME, "model")
        if not success:
            st.error("Failed to download model")
            return None, None, None
    
    # Load vocabulary
    vocab_data = torch.load(VOCAB_FILE_NAME, map_location='cpu')
    if isinstance(vocab_data, dict):
        word_to_idx = vocab_data.get('word_to_idx', vocab_data)
    else:
        word_to_idx = vocab_data
    vocab_size = len(word_to_idx)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_enc = create_text_encoder('bilstm', vocab_size, hidden=128)
    model = MultimodalModel('mobilenet_v2', text_enc)
    
    checkpoint = torch.load(MODEL_FILE_NAME, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    st.success("✅ Model and vocabulary loaded successfully!")
    
    return model, word_to_idx, device

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Load model
model, word_to_idx, device = load_model_and_vocab()

if model is None:
    st.error("❌ Failed to load model. Please check your Google Drive files.")
    st.stop()

# ============================================================================
# LAYOUT - TWO COLUMNS
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Drawing")
    
    uploaded_file = st.file_uploader(
        "Upload a drawing (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Drawing", use_container_width=True)
    else:
        st.info("Upload a drawing (JPG/PNG)")
        image = None
    
    st.subheader("✍️ Self-Reflection Text")
    
    text_input = st.text_area(
        "Enter the child's self-reflection about their drawing:",
        placeholder="Example: I drew this because I felt very happy today...",
        height=100,
        label_visibility="collapsed"
    )
    if not text_input:
        st.caption("Enter the child's self-reflection text")

with col2:
    st.subheader("📊 Analysis Results")
    
    analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    if analyze_button and uploaded_file is not None:
        with st.spinner("Analyzing..."):
            try:
                # Preprocess inputs
                def preprocess_image(img):
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
                    ])
                    return transform(img).unsqueeze(0)
                
                def preprocess_text(text, word_to_idx, max_len=50):
                    tokens = text.lower().split()
                    token_ids = []
                    for token in tokens:
                        token_ids.append(word_to_idx.get(token, word_to_idx.get('<UNK>', 1)))
                    if len(token_ids) > max_len:
                        token_ids = token_ids[:max_len]
                    else:
                        token_ids = token_ids + [0] * (max_len - len(token_ids))
                    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
                
                image_tensor = preprocess_image(image)
                text_tensor = preprocess_text(text_input, word_to_idx, max_len=50)
                
                # Make prediction
                with torch.no_grad():
                    image_tensor = image_tensor.to(device)
                    text_tensor = text_tensor.to(device)
                    outputs = model(image_tensor, text_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                class_names = ['Happy', 'Sad']
                predicted_class = class_names[prediction]
                
                confidence_pct = confidence * 100
                happy_pct = probabilities[0][0].item() * 100
                sad_pct = probabilities[0][1].item() * 100
                
                # Display prediction
                if predicted_class == 'Happy':
                    st.markdown(f"""
                    <div class="result-box happy">
                        <h2 style="margin: 0;">😊 Happy</h2>
                        <p style="font-size: 1.2rem; margin: 0;">Confidence: {confidence_pct:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box sad">
                        <h2 style="margin: 0;">😢 Sad</h2>
                        <p style="font-size: 1.2rem; margin: 0;">Confidence: {confidence_pct:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence bars
                st.markdown("**Confidence Distribution**")
                col_h, col_s = st.columns(2)
                with col_h:
                    st.write("Happy")
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill happy" style="width: {happy_pct:.1f}%;">
                            {happy_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_s:
                    st.write("Sad")
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill sad" style="width: {sad_pct:.1f}%;">
                            {sad_pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if confidence_pct < 85:
                    st.warning("⚠️ Low confidence - Manual review recommended")
                else:
                    st.success("✅ High confidence prediction")
                
                # ================================================================
                # L32 GRAD-CAM - Visual Explanation
                # ================================================================
                st.markdown("---")
                st.markdown("### 🔍 L32 Layer-Wise Grad-CAM")
                st.caption("Layer 32 (L32) captures high-level emotion-relevant regions")
                
                overlay, heatmap = generate_l32_gradcam(
                    model, image_tensor, text_tensor, device, target_class=prediction
                )
                
                if overlay is not None and heatmap is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    
                    # Original image
                    axes[0].imshow(image.resize((224, 224)))
                    axes[0].set_title("Original Drawing")
                    axes[0].axis('off')
                    
                    # Heatmap only
                    axes[1].imshow(heatmap, cmap='jet')
                    axes[1].set_title("L32 Heatmap")
                    axes[1].axis('off')
                    
                    # Overlay
                    axes[2].imshow(overlay)
                    axes[2].set_title("L32 Grad-CAM Overlay")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.caption("🟡 Yellow/Red areas = Regions most important for the prediction")
                    st.info("**L32** captures high-level concepts and complex patterns that are most relevant for emotion recognition.")
                else:
                    st.info("L32 Grad-CAM explanation not available")
                
                # ================================================================
                # LIME - Text Explanation
                # ================================================================
                st.markdown("### 📝 LIME: Text Explanation")
                
                if text_input.strip():
                    base_pred, word_importance = generate_lime_text_explanation(
                        text_input, model, word_to_idx, device, max_len=50
                    )
                    
                    if word_importance and len(word_importance) > 0:
                        highlighted_words = []
                        for word, importance in word_importance:
                            if importance > 0.7:
                                cls = "high"
                            elif importance > 0.4:
                                cls = "medium"
                            else:
                                cls = "low"
                            highlighted_words.append(f'<span class="word-highlight {cls}">{word}</span>')
                        
                        st.markdown(
                            f'<div style="padding: 10px; background: #f8f9fa; border-radius: 8px; font-size: 0.95rem; line-height: 1.8;">'
                            f'{" ".join(highlighted_words)}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        st.caption("🔴 High importance | 🟡 Medium | ⚪ Low")
                        class_names_expl = ['Happy', 'Sad']
                        st.caption(f"Text-based prediction: {class_names_expl[base_pred]}")
                    else:
                        st.info("LIME explanation not available for this text")
                else:
                    st.info("No text provided for LIME explanation")
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif analyze_button:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a drawing")
        if not text_input.strip():
            st.warning("⚠️ Please enter self-reflection text")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.7rem;">
    MobileNetV2 + BiLSTM | L32 Layer-Wise Grad-CAM + LIME | 92.69% Val Accuracy | KIDO Dataset
</div>
""", unsafe_allow_html=True)
