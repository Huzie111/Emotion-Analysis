"""
Multimodal Emotion Classification System
Deployed on Streamlit Cloud
Model: MobileNetV2 + BiLSTM (92.69% Validation Accuracy)
Explainability: Layer 5 Grad-CAM for Visual + LIME for Text
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import gdown
import os
import requests
import re
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GOOGLE DRIVE FILE IDs
# ============================================================================

MODEL_FILE_ID = "11lYY2-0tXlF4mE1peB2ReQy9bMLlp2mp"
VOCAB_FILE_ID = "1r2mCVi-tVjeI18P2dBFFdlYAeHNuKnm-"

MODEL_FILE_NAME = "MM_MobileNetV2_BiLSTM_final.pt"
VOCAB_FILE_NAME = "vocabulary.pth"

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

def get_mobilenetv2_encoder(pretrained=False):
    """Get MobileNetV2 encoder matching saved architecture."""
    model = models.mobilenet_v2(pretrained=pretrained)
    
    class MobileNetV2Encoder(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 1280
            
        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return x
    
    return MobileNetV2Encoder(model), 1280

def get_vision_encoder(name, pretrained=False):
    if name == 'mobilenet_v2':
        return get_mobilenetv2_encoder(pretrained)
    
    backbones = {
        'efficientnet_b0': (models.efficientnet_b0, 1280),
        'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, 1024),
    }
    
    if name in backbones:
        model_fn, dim = backbones[name]
        model = model_fn(pretrained=pretrained)
        if hasattr(model, 'classifier'):
            model.classifier = nn.Identity()
        elif hasattr(model, 'fc'):
            model.fc = nn.Identity()
        return model, dim
    
    raise ValueError(f"Unknown model: {name}")

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
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_file_gdown(file_id, file_name, description):
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)
        if os.path.exists(file_name):
            return True
        return False
    except Exception as e:
        st.warning(f"gdown failed: {e}")
        return False

def download_file_requests(file_id, file_name, description):
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
    except Exception as e:
        st.warning(f"Requests failed: {e}")
        return False

def download_file(file_id, file_name, description):
    if download_file_gdown(file_id, file_name, description):
        return True
    if download_file_requests(file_id, file_name, description):
        return True
    st.error(f"All download methods failed for {description}")
    return False

# ============================================================================
# LOAD FUNCTIONS
# ============================================================================

def load_vocabulary():
    try:
        if not os.path.exists(VOCAB_FILE_NAME):
            success = download_file(VOCAB_FILE_ID, VOCAB_FILE_NAME, "vocabulary")
            if not success:
                return None, None
        
        file_size = os.path.getsize(VOCAB_FILE_NAME)
        if file_size < 1000:
            os.remove(VOCAB_FILE_NAME)
            success = download_file(VOCAB_FILE_ID, VOCAB_FILE_NAME, "vocabulary")
            if not success:
                return None, None
        
        vocab_data = torch.load(VOCAB_FILE_NAME, map_location='cpu')
        
        if isinstance(vocab_data, dict):
            word_to_idx = vocab_data.get('word_to_idx', vocab_data)
        else:
            word_to_idx = vocab_data
        
        vocab_size = len(word_to_idx)
        return word_to_idx, vocab_size
        
    except Exception as e:
        st.error(f"Error loading vocabulary: {e}")
        return None, None

def load_model(vocab_size):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(MODEL_FILE_NAME):
            success = download_file(MODEL_FILE_ID, MODEL_FILE_NAME, "model")
            if not success:
                return None, device
        
        file_size = os.path.getsize(MODEL_FILE_NAME)
        if file_size < 1000000:
            os.remove(MODEL_FILE_NAME)
            success = download_file(MODEL_FILE_ID, MODEL_FILE_NAME, "model")
            if not success:
                return None, device
        
        # Create model
        text_enc = create_text_encoder('bilstm', vocab_size, hidden=128)
        model = MultimodalModel('mobilenet_v2', text_enc)
        
        # Load weights
        checkpoint = torch.load(MODEL_FILE_NAME, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle any mismatches
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
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

def predict(model, image, text_tensor, device):
    with torch.no_grad():
        image = image.to(device)
        text_tensor = text_tensor.to(device)
        outputs = model(image, text_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    return prediction, confidence, probabilities

# ============================================================================
# LAYER 5 GRAD-CAM
# ============================================================================

class GradCAM:
    """Grad-CAM implementation for target layer."""
    
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
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
        """Generate Grad-CAM heatmap."""
        
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        image = image.to(self.device)
        text_tensor = text_tensor.to(self.device)
        image.requires_grad = True
        
        output = self.model(image, text_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Get gradients and activations
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return None, target_class
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        heatmap = cam.squeeze().cpu().detach().numpy()
        
        return heatmap, target_class
    
    def overlay_heatmap(self, image, heatmap, alpha=0.5):
        """Overlay heatmap on original image."""
        
        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            img = image.squeeze().cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            img = np.uint8(255 * img)
        else:
            img = np.array(image)
        
        # Ensure img is RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert to heatmap color
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        
        # Overlay
        overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        
        return overlayed, heatmap_resized

def get_layer_5(model):
    """
    Get Layer 5 (the 5th Conv2d layer) for Grad-CAM.
    Layer 5 captures: edges, colors, and basic patterns.
    """
    
    # Collect all Conv2d layers from vision encoder
    conv_layers = []
    
    def collect_layers(module):
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                conv_layers.append(child)
            collect_layers(child)
    
    collect_layers(model.vision)
    
    st.info(f"Found {len(conv_layers)} Conv2d layers in vision encoder")
    
    # Return layer at index 5 (0-based)
    target_index = 5
    if target_index < len(conv_layers):
        st.info(f"Using Layer {target_index}: {type(conv_layers[target_index]).__name__}")
        return conv_layers[target_index]
    else:
        # Fallback to last layer if index 5 doesn't exist
        st.warning(f"Layer {target_index} not found. Using last layer (index {len(conv_layers)-1}).")
        return conv_layers[-1]

def generate_layer5_gradcam(model, image, text_tensor, device, target_class=None):
    """Generate Layer 5 Grad-CAM explanation."""
    try:
        # Get Layer 5
        target_layer = get_layer_5(model)
        
        if target_layer is None:
            return None, None
        
        # Create Grad-CAM instance
        gradcam = GradCAM(model, target_layer, device)
        
        # Generate heatmap
        heatmap, target_class = gradcam.generate_heatmap(
            image, text_tensor, target_class
        )
        
        if heatmap is None:
            return None, None
        
        # Create overlay
        overlay, heatmap_resized = gradcam.overlay_heatmap(
            image, heatmap, alpha=0.6
        )
        
        return overlay, heatmap_resized
        
    except Exception as e:
        st.warning(f"Layer 5 Grad-CAM error: {e}")
        return None, None

# ============================================================================
# LIME FOR TEXT EXPLANATION
# ============================================================================

def generate_lime_text_explanation(text, model, word_to_idx, device, max_len=50):
    words = text.lower().split()
    if len(words) == 0:
        return None, []
    
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
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Emotion Analysis from Children's Drawings",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #2c3e50; text-align: center; margin-bottom: 0.3rem; }
    .sub-header { font-size: 1rem; color: #7f8c8d; text-align: center; margin-bottom: 1.5rem; }
    .result-box { padding: 15px; border-radius: 10px; text-align: center; margin: 5px 0; }
    .happy { background-color: #d4edda; border: 2px solid #28a745; }
    .sad { background-color: #f8d7da; border: 2px solid #dc3545; }
    .confidence-bar { height: 16px; background: #e9ecef; border-radius: 8px; overflow: hidden; margin: 5px 0; }
    .confidence-fill { height: 100%; border-radius: 8px; transition: width 0.5s; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.6rem; font-weight: bold; }
    .confidence-fill.happy { background: linear-gradient(90deg, #28a745, #20c997); }
    .confidence-fill.sad { background: linear-gradient(90deg, #dc3545, #e74c3c); }
    .stButton button { width: 100%; background: #3498db; color: white; font-weight: 600; padding: 10px; font-size: 1rem; }
    .stButton button:hover { background: #2980b9; }
    .model-info { background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; margin-bottom: 10px; font-size: 0.8rem; }
    .model-info table { width: 100%; font-size: 0.8rem; }
    .model-info td { padding: 2px 6px; }
    .model-info .label { font-weight: 600; color: #495057; }
    .word-highlight { display: inline-block; padding: 2px 6px; margin: 1px; border-radius: 4px; font-size: 0.9rem; }
    .word-highlight.high { background: #dc3545; color: white; }
    .word-highlight.medium { background: #ffc107; color: #333; }
    .word-highlight.low { background: #e9ecef; color: #333; }
    .layer-info { background: #e8f4fd; padding: 8px 12px; border-radius: 6px; border-left: 4px solid #3498db; font-size: 0.8rem; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🎨 Emotion Analysis from Children\'s Drawings</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MobileNetV2 + BiLSTM | Layer 5 Grad-CAM + LIME Explainability</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - LOAD RESOURCES
# ============================================================================

with st.sidebar:
    st.markdown("### Model Info")
    st.markdown("""
    <div class="model-info">
        <table>
            <tr><td class="label">Model</td><td>MobileNetV2+BiLSTM</td></tr>
            <tr><td class="label">Task</td><td>Happy vs Sad</td></tr>
            <tr><td class="label">Val Accuracy</td><td>92.69%</td></tr>
            <tr><td class="label">Test Accuracy</td><td>92.00%</td></tr>
            <tr><td class="label">Size</td><td>19.36 MB</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Explainability")
    st.markdown("""
    <div class="model-info">
        <table>
            <tr><td class="label">Visual</td><td>Layer 5 Grad-CAM</td></tr>
            <tr><td class="label">Text</td><td>LIME</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="layer-info">
        <b>Layer 5</b> captures early visual features:
        <br>• Edges and contours
        <br>• Colors and textures
        <br>• Simple patterns
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Clinical Disclaimer:** Screening only. Confidence < 85% → Manual review.")
    
    st.markdown("---")
    st.markdown("### Loading Resources...")
    
    word_to_idx = None
    vocab_size = 3423
    model = None
    device = torch.device('cpu')
    
    try:
        result = load_vocabulary()
        if result is not None and len(result) == 2:
            word_to_idx, vocab_size = result
            if word_to_idx is not None:
                st.success(f"Vocabulary loaded (Size: {vocab_size})")
            else:
                st.warning("Using fallback vocabulary")
                word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        else:
            st.warning("Using fallback vocabulary")
            word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    except Exception as e:
        st.error(f"Vocabulary error: {e}")
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    
    if word_to_idx is not None:
        try:
            model, device = load_model(vocab_size)
            if model is not None:
                st.success("Model ready!")
            else:
                st.error("Model not loaded")
        except Exception as e:
            st.error(f"Model error: {e}")

# ============================================================================
# MAIN CONTENT - TWO COLUMNS
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Drawing")
    
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
    
    st.subheader("Self-Reflection Text")
    
    text_input = st.text_area(
        "Enter the child's self-reflection about their drawing:",
        placeholder="Example: I drew this because I felt very happy today...",
        height=100,
        label_visibility="collapsed"
    )
    if not text_input:
        st.caption("Enter the child's self-reflection text")

with col2:
    st.subheader("Analysis Results")
    
    analyze_button = st.button("Analyze Emotion", type="primary", use_container_width=True)
    
    if analyze_button and uploaded_file is not None:
        if model is None:
            st.error("Model not loaded. Please check the sidebar.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    # Preprocess inputs
                    image_tensor = preprocess_image(image)
                    text_tensor = preprocess_text(text_input, word_to_idx, max_len=50)
                    
                    # Make prediction
                    prediction, confidence, probabilities = predict(
                        model, image_tensor, text_tensor, device
                    )
                    
                    class_names = ['Happy', 'Sad']
                    predicted_class = class_names[prediction]
                    
                    confidence_pct = confidence * 100
                    happy_pct = probabilities[0][0].item() * 100
                    sad_pct = probabilities[0][1].item() * 100
                    
                    # Display prediction
                    if predicted_class == 'Happy':
                        st.markdown(f"""
                        <div class="result-box happy">
                            <h2 style="margin: 0;">Happy</h2>
                            <p style="font-size: 1.2rem; margin: 0;">Confidence: {confidence_pct:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box sad">
                            <h2 style="margin: 0;">Sad</h2>
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
                        st.warning("Low confidence - Manual review recommended")
                    else:
                        st.success("High confidence prediction")
                    
                    # ========================================================
                    # LAYER 5 GRAD-CAM
                    # ========================================================
                    st.markdown("---")
                    st.markdown("### Layer 5 Grad-CAM: Visual Attention")
                    st.caption("Layer 5 captures: Edges, Colors, and Basic Patterns")
                    
                    overlay, heatmap = generate_layer5_gradcam(
                        model, image_tensor, text_tensor, device, target_class=prediction
                    )
                    
                    if overlay is not None and heatmap is not None:
                        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                        
                        # Original image
                        axes[0].imshow(image.resize((224, 224)))
                        axes[0].set_title("Original")
                        axes[0].axis('off')
                        
                        # Heatmap only
                        axes[1].imshow(heatmap, cmap='jet')
                        axes[1].set_title("Layer 5 Heatmap")
                        axes[1].axis('off')
                        
                        # Overlay
                        axes[2].imshow(overlay)
                        axes[2].set_title("Grad-CAM Overlay")
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        st.caption("🟡 Yellow/Red areas = Regions most important for the prediction")
                        st.info("Layer 5 captures early visual features like edges, colors, and simple patterns that form the building blocks of the drawing.")
                    else:
                        st.info("Layer 5 Grad-CAM explanation not available")
                    
                    # ========================================================
                    # LIME FOR TEXT
                    # ========================================================
                    st.markdown("### LIME: Text Explanation")
                    
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
                            
                            st.caption("High importance = Red | Medium = Yellow | Low = Gray")
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
            st.warning("Please upload a drawing")
        if not text_input.strip():
            st.warning("Please enter self-reflection text")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.7rem;">
    MobileNetV2 + BiLSTM | Layer 5 Grad-CAM + LIME | 92.69% Val Accuracy | KIDO Dataset
</div>
""", unsafe_allow_html=True)
