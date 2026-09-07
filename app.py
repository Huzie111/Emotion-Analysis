"""
Multimodal Emotion Classification System
Deployed on Streamlit Cloud
Model: MobileNetV2 + BiLSTM (92.69% Validation Accuracy)
Explainability: L32 Grad-CAM (Layer 32) for Visual + LIME for Text
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import gdown
import os
import requests
import re
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GOOGLE DRIVE FILE IDs - UPDATED
# ============================================================================

# Model: MM_MobileNetV2_BiLSTM
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
# DOWNLOAD FUNCTIONS
# ============================================================================

def download_file_gdown(file_id, file_name, description):
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)
        return True
    except Exception as e:
        st.warning(f"gdown failed for {description}: {e}")
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
        st.warning(f"Requests failed for {description}: {e}")
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
            with st.spinner("Downloading vocabulary..."):
                success = download_file(VOCAB_FILE_ID, VOCAB_FILE_NAME, "vocabulary")
                if not success:
                    return None, None
        
        file_size = os.path.getsize(VOCAB_FILE_NAME)
        if file_size < 1000:
            st.warning(f"Vocabulary file too small ({file_size} bytes). Re-downloading...")
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
            with st.spinner("Downloading model (this may take a few minutes)..."):
                success = download_file(MODEL_FILE_ID, MODEL_FILE_NAME, "model")
                if not success:
                    return None, device
        
        file_size = os.path.getsize(MODEL_FILE_NAME)
        if file_size < 1000000:
            st.warning(f"Model file too small ({file_size/1024:.1f} KB). Re-downloading...")
            os.remove(MODEL_FILE_NAME)
            success = download_file(MODEL_FILE_ID, MODEL_FILE_NAME, "model")
            if not success:
                return None, device
        
        text_enc = create_text_encoder('bilstm', vocab_size, hidden=128)
        model = MultimodalModel('mobilenet_v2', text_enc)
        
        checkpoint = torch.load(MODEL_FILE_NAME, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
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

def tokenize_text(text, word_to_idx, max_len=50):
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
# L32 GRAD-CAM (Layer 32) - MobileNetV2 specific
# ============================================================================

def get_mobilenetv2_target_layer(model):
    """
    Get Layer 32 (L32) of MobileNetV2 for Grad-CAM.
    MobileNetV2 features are sequential. Layer 32 corresponds to the
    last convolutional layer before the final pooling.
    """
    if hasattr(model.vision, 'features'):
        # Get the last layer of features (L32 in MobileNetV2 terminology)
        target_layer = model.vision.features[-1]
        return target_layer
    else:
        # Fallback: use the last conv layer
        last_conv = None
        for name, module in model.vision.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

def upsample_heatmap(heatmap, target_size):
    """Upsample heatmap using PyTorch interpolate."""
    import torch.nn.functional as F
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    upsampled = F.interpolate(heatmap_tensor, size=target_size, mode='bilinear', align_corners=False)
    return upsampled.squeeze().cpu().numpy()

def generate_l32_gradcam(model, image, text_tensor, device, target_class=None):
    """
    Generate L32 Grad-CAM for MobileNetV2.
    Uses Layer 32 (the last conv layer) as the target.
    """
    try:
        # Get L32 target layer (last conv layer of MobileNetV2)
        target_layer = get_mobilenetv2_target_layer(model)
        
        if target_layer is None:
            return None
        
        # Register forward hook to capture activations
        activations = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0]
        
        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        # Forward pass
        image = image.to(device)
        text_tensor = text_tensor.to(device)
        
        # Enable gradient for Grad-CAM
        image.requires_grad = True
        
        # Forward through the model
        outputs = model(image, text_tensor)
        
        # Get target class (if not specified, use predicted class)
        if target_class is None:
            target_class = torch.argmax(outputs, dim=1).item()
        
        # Backward pass for target class
        model.zero_grad()
        outputs[0][target_class].backward()
        
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()
        
        # Generate heatmap from activations and gradients
        if activations is not None and gradients is not None:
            # Global average pooling of gradients
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            
            # Weighted combination of activation maps
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]
            
            # Average over channels
            heatmap = torch.mean(activations, dim=1).squeeze().cpu().detach().numpy()
            
            # ReLU and normalize
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            # Upsample to input size
            heatmap_resized = upsample_heatmap(heatmap, (224, 224))
            return heatmap_resized
        
        return None
        
    except Exception as e:
        print(f"L32 Grad-CAM error: {e}")
        return None

# ============================================================================
# LIME FOR TEXT EXPLANATION
# ============================================================================

def generate_lime_text_explanation(text, model, word_to_idx, device, max_len=50):
    """
    Generate LIME-like explanation for text using perturbation-based approach.
    """
    words = text.lower().split()
    if len(words) == 0:
        return None, []
    
    # Get base prediction for the original text
    text_tensor = tokenize_text(text, word_to_idx, max_len)
    text_tensor = text_tensor.to(device)
    
    # Create a dummy image tensor (black image) for text-only explanation
    dummy_image = torch.zeros(1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(dummy_image, text_tensor)
        base_probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        base_pred = np.argmax(base_probs)
    
    # Perturb words and measure impact
    word_importance = []
    
    for i, word in enumerate(words):
        # Remove word and measure change in prediction
        perturbed_words = words[:i] + words[i+1:]
        perturbed_text = ' '.join(perturbed_words)
        
        if len(perturbed_text.strip()) == 0:
            continue
        
        perturbed_tensor = tokenize_text(perturbed_text, word_to_idx, max_len)
        perturbed_tensor = perturbed_tensor.to(device)
        
        with torch.no_grad():
            perturbed_outputs = model(dummy_image, perturbed_tensor)
            perturbed_probs = torch.softmax(perturbed_outputs, dim=1).cpu().numpy()[0]
        
        # Calculate importance as change in confidence for the base prediction
        importance = abs(base_probs[base_pred] - perturbed_probs[base_pred])
        word_importance.append((word, importance))
    
    # Normalize importance scores
    if len(word_importance) > 0:
        max_imp = max([imp for _, imp in word_importance])
        if max_imp > 0:
            word_importance = [(w, imp / max_imp) for w, imp in word_importance]
    
    return base_pred, word_importance

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #2c3e50; text-align: center; margin-bottom: 0.3rem; }
    .sub-header { font-size: 0.9rem; color: #7f8c8d; text-align: center; margin-bottom: 1rem; }
    .result-box { padding: 15px; border-radius: 10px; text-align: center; margin: 5px 0; }
    .happy { background-color: #d4edda; border: 2px solid #28a745; }
    .sad { background-color: #f8d7da; border: 2px solid #dc3545; }
    .confidence-bar { height: 16px; background: #e9ecef; border-radius: 8px; overflow: hidden; margin: 5px 0; }
    .confidence-fill { height: 100%; border-radius: 8px; transition: width 0.5s; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.6rem; font-weight: bold; }
    .confidence-fill.happy { background: linear-gradient(90deg, #28a745, #20c997); }
    .confidence-fill.sad { background: linear-gradient(90deg, #dc3545, #e74c3c); }
    .stButton button { width: 100%; background: #3498db; color: white; font-weight: 600; padding: 8px; font-size: 0.9rem; }
    .stButton button:hover { background: #2980b9; }
    .model-info { background: #f8f9fa; padding: 10px; border-radius: 8px; border: 1px solid #e9ecef; margin-bottom: 10px; font-size: 0.8rem; }
    .model-info table { width: 100%; font-size: 0.8rem; }
    .model-info td { padding: 2px 6px; }
    .model-info .label { font-weight: 600; color: #495057; }
    .xai-header { font-size: 1rem; font-weight: 600; color: #2c3e50; margin: 10px 0 5px 0; }
    .word-highlight { display: inline-block; padding: 2px 6px; margin: 1px; border-radius: 4px; font-size: 0.9rem; transition: all 0.3s; }
    .word-highlight.high { background: #dc3545; color: white; }
    .word-highlight.medium { background: #ffc107; color: #333; }
    .word-highlight.low { background: #e9ecef; color: #333; }
    .explanation-caption { font-size: 0.7rem; color: #7f8c8d; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
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
            <tr><td class="label">Visual</td><td>L32 Grad-CAM</td></tr>
            <tr><td class="label">Text</td><td>LIME</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Clinical Disclaimer:** Screening only. Confidence < 85% → Manual review.")
    
    # Load resources
    st.markdown("---")
    st.markdown("Loading resources...")
    
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
        st.warning("Using fallback vocabulary")
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    
    if word_to_idx is not None:
        try:
            model, device = load_model(vocab_size)
            if model is not None:
                st.success("Model ready!")
            else:
                st.error("Model not loaded")
                st.info("Make sure Google Drive files are publicly accessible:")
                st.code(f"Model ID: {MODEL_FILE_ID}")
                st.code(f"Vocab ID: {VOCAB_FILE_ID}")
        except Exception as e:
            st.error(f"Model error: {e}")
    else:
        st.error("Cannot load model without vocabulary")

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<div class="main-header">Multimodal Emotion Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MobileNetV2 + BiLSTM | Children\'s Drawing + Self-Reflection Analysis</div>', unsafe_allow_html=True)

# Compact 2-column layout
col1, col2 = st.columns([1, 1])

with col1:
    # Image upload
    uploaded_image = st.file_uploader("Upload Drawing", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, caption="Uploaded Drawing", use_container_width=True)
    else:
        st.info("Upload a drawing (JPG/PNG)")
        image = None
    
    # Text input
    text_input = st.text_area(
        "Self-Reflection Text",
        placeholder="e.g., I felt happy when I played with my friends today...",
        height=80,
        label_visibility="collapsed"
    )
    if not text_input:
        st.caption("Enter the child's self-reflection text")

with col2:
    # Results area
    analyze_button = st.button("Analyze Emotion", type="primary", use_container_width=True)
    
    if analyze_button and uploaded_image is not None and text_input.strip():
        if model is None:
            st.error("Model not loaded. Please check:")
            st.info("1. Google Drive files are publicly shared")
            st.info("2. File IDs are correct")
            st.code(f"Model ID: {MODEL_FILE_ID}")
            st.code(f"Vocab ID: {VOCAB_FILE_ID}")
        else:
            with st.spinner("Analyzing..."):
                try:
                    image_tensor = preprocess_image(image)
                    text_tensor = tokenize_text(text_input, word_to_idx, max_len=50)
                    prediction, confidence, probabilities = predict(model, image_tensor, text_tensor, device)
                    
                    class_names = ['Happy', 'Sad']
                    predicted_class = class_names[prediction]
                    
                    confidence_pct = confidence * 100
                    happy_pct = probabilities[0][0].item() * 100
                    sad_pct = probabilities[0][1].item() * 100
                    
                    # Show prediction
                    if predicted_class == 'Happy':
                        st.markdown(f"""
                        <div class="result-box happy">
                            <h2 style="margin: 0;">Happy</h2>
                            <p style="font-size: 1rem; margin: 0;">Confidence: {confidence_pct:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box sad">
                            <h2 style="margin: 0;">Sad</h2>
                            <p style="font-size: 1rem; margin: 0;">Confidence: {confidence_pct:.1f}%</p>
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
                    
                    # =========================================================
                    # L32 GRAD-CAM - Visual Explanation
                    # =========================================================
                    st.markdown("---")
                    st.markdown("### L32 Grad-CAM: Visual Attention")
                    st.caption("Layer 32 - Last convolutional layer of MobileNetV2")
                    
                    heatmap = generate_l32_gradcam(model, image_tensor, text_tensor, device, target_class=prediction)
                    
                    if heatmap is not None:
                        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                        
                        # Original image
                        axes[0].imshow(image.resize((224, 224)))
                        axes[0].set_title("Original")
                        axes[0].axis('off')
                        
                        # Heatmap only
                        axes[1].imshow(heatmap, cmap='jet')
                        axes[1].set_title("L32 Grad-CAM")
                        axes[1].axis('off')
                        
                        # Overlay
                        axes[2].imshow(image.resize((224, 224)))
                        axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
                        axes[2].set_title("Overlay")
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("L32 Grad-CAM explanation not available")
                    
                    # =========================================================
                    # LIME - Text Explanation
                    # =========================================================
                    st.markdown("### LIME: Text Explanation")
                    
                    # Generate LIME explanation
                    base_pred, word_importance = generate_lime_text_explanation(
                        text_input, model, word_to_idx, device, max_len=50
                    )
                    
                    if word_importance and len(word_importance) > 0:
                        # Display highlighted words
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
                        
                        # Show class prediction for explanation
                        class_names_expl = ['Happy', 'Sad']
                        st.caption(f"Prediction based on text: {class_names_expl[base_pred]}")
                    else:
                        st.info("LIME explanation not available for this text")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    elif analyze_button:
        if uploaded_image is None:
            st.warning("Upload a drawing")
        if not text_input.strip():
            st.warning("Enter text")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #95a5a6; font-size: 0.7rem;">
    MobileNetV2 + BiLSTM | L32 Grad-CAM + LIME | 92.69% Val Accuracy | KIDO Dataset
</div>
""", unsafe_allow_html=True)
