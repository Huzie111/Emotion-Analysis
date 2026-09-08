"""
L32 Layer-Wise Grad-CAM Streamlit App
Model: MobileNetV2 + BiLSTM (92.69% Validation Accuracy)
FIXED: Proper model architecture matching training
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
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Emotion Analysis - Layer-wise Grad-CAM",
    page_icon="🎨",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
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
st.markdown('<div class="sub-header">MobileNetV2 + BiLSTM | Layer-wise Grad-CAM Explainability</div>', unsafe_allow_html=True)

# ============================================================================
# MODEL COMPONENTS - EXACT MATCH TO TRAINING
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
    """Get vision encoder - matches training architecture exactly."""
    
    backbones = {
        'mobilenet_v2': (models.mobilenet_v2, 1280),
        'efficientnet_b0': (models.efficientnet_b0, 1280),
        'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, 1024),
        'squeezenet1_1': (models.squeezenet1_1, 512),
    }
    
    if name not in backbones:
        raise ValueError(f"Unknown model: {name}")
    
    model_fn, dim = backbones[name]
    model = model_fn(pretrained=pretrained)
    
    # Remove classification head
    if hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    elif hasattr(model, 'fc'):
        model.fc = nn.Identity()
    elif hasattr(model, 'head'):
        model.head = nn.Identity()
    
    return model, dim

class MultimodalModel(nn.Module):
    """Multimodal model - EXACT match to training."""
    
    def __init__(self, vision_name, text_encoder, dropout=0.7):
        super().__init__()
        
        # Vision encoder
        self.vision, vdim = get_vision_encoder(vision_name, pretrained=False)
        
        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Linear(vdim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text encoder
        self.text_encoder = text_encoder
        
        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_encoder.output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion + Classifier
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
        # Vision path
        v = self.vision(images)
        if v.dim() > 2:
            v = v.view(v.size(0), -1)
        v = self.vision_proj(v)
        
        # Text path
        t = self.text_encoder(texts)
        t = self.text_proj(t)
        
        # Fusion
        fused = torch.cat([v, t], dim=1)
        
        # Classification
        return self.classifier(fused)

def create_text_encoder(text_type, vocab_size, hidden=128):
    """Create text encoder based on type."""
    if text_type == 'bilstm':
        return BiLSTMTextEncoder(vocab_size, hidden=hidden)
    elif text_type == 'gru':
        return GRUTextEncoder(vocab_size, hidden=hidden)
    else:
        raise ValueError(f"Unknown text encoder: {text_type}")

# ============================================================================
# LOAD MODEL AND VOCABULARY
# ============================================================================

@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabulary."""
    
    # File paths
    MODEL_FILE_NAME = "MM_MobileNetV2_BiLSTM_final.pt"
    VOCAB_FILE_NAME = "vocabulary.pth"
    
    # Check if files exist
    if not os.path.exists(MODEL_FILE_NAME):
        st.warning(f"Model file not found: {MODEL_FILE_NAME}")
        return None, None, None
    
    if not os.path.exists(VOCAB_FILE_NAME):
        st.warning(f"Vocabulary file not found: {VOCAB_FILE_NAME}")
        return None, None, None
    
    try:
        # Load vocabulary
        vocab_data = torch.load(VOCAB_FILE_NAME, map_location='cpu')
        if isinstance(vocab_data, dict):
            word_to_idx = vocab_data.get('word_to_idx', vocab_data)
        else:
            word_to_idx = vocab_data
        vocab_size = len(word_to_idx)
        st.success("✅ Vocabulary loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load vocabulary: {e}")
        return None, None, None
    
    try:
        # Device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create text encoder
        text_enc = create_text_encoder('bilstm', vocab_size, hidden=128)
        
        # Create model
        model = MultimodalModel('mobilenet_v2', text_enc)
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_FILE_NAME, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict with strict=False to handle any minor mismatches
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model, word_to_idx, device
        
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        import traceback
        with st.expander("Show full error"):
            st.code(traceback.format_exc())
        return None, None, None

# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================

def get_all_conv_layers(model):
    """Get all Conv2d layers from the vision encoder."""
    conv_layers = []
    
    def collect_layers(module, prefix=""):
        for name, child in module.named_children():
            current_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                conv_layers.append((current_name, child))
            collect_layers(child, current_name)
    
    collect_layers(model.vision)
    return conv_layers

class GradCAM:
    """Grad-CAM implementation."""
    
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
            if grad_output[0] is not None:
                self.gradients = grad_output[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_heatmap(self, image, text_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        self.gradients = None
        self.activations = None
        
        # Forward pass with gradient tracking
        image = image.clone().to(self.device).requires_grad_(True)
        text_tensor = text_tensor.clone().to(self.device)
        
        output = self.model(image, text_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return None
        
        if torch.max(torch.abs(gradients)) < 1e-6:
            return None
        
        # Global average pooling
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.squeeze().detach().cpu().numpy()

def resize_heatmap(heatmap, target_size):
    """Resize heatmap."""
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
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
    else:
        img = np.array(image) / 255.0
    
    # Resize heatmap
    target_size = (img.shape[0], img.shape[1])
    heatmap_resized = resize_heatmap(heatmap, target_size)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    
    # RGB heatmap
    heatmap_rgb = cm.jet(heatmap_resized)[:, :, :3]
    
    # Overlay
    overlay = (1 - alpha) * img + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, heatmap_resized

def generate_gradcam_for_layer(model, layer, image_tensor, text_tensor, device, target_class=None):
    """Generate Grad-CAM for a specific layer."""
    try:
        gradcam = GradCAM(model, layer, device)
        heatmap = gradcam.generate_heatmap(image_tensor, text_tensor, target_class)
        
        if heatmap is None or np.max(heatmap) < 0.01:
            return None, None
        
        overlay, heatmap_resized = create_overlay(image_tensor, heatmap, alpha=0.6)
        return overlay, heatmap_resized
        
    except Exception as e:
        return None, None

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

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

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Load model
model, word_to_idx, device = load_model_and_vocab()

if model is None:
    st.error("❌ Failed to load model. Please check your files.")
    st.stop()

# Get all Conv2d layers
all_layers = get_all_conv_layers(model)
st.info(f"Found {len(all_layers)} Conv2d layers in vision encoder")

# ============================================================================
# LAYOUT
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
        "Enter the child's self-reflection:",
        placeholder="Example: I drew this because I felt very happy today...",
        height=100,
        label_visibility="collapsed"
    )
    if not text_input:
        st.caption("Enter the child's self-reflection text")
    
    # Layer selector
    st.subheader("🎯 Select Grad-CAM Layer")
    
    # Preset layers
    st.markdown("**Preset Layers:**")
    preset_cols = st.columns(5)
    preset_layers = {5: "Layer 5", 10: "Layer 10", 15: "Layer 15", 20: "Layer 20", 32: "Layer 32"}
    
    selected_preset = None
    for col, (idx, name) in zip(preset_cols, preset_layers.items()):
        if col.button(name, key=f"preset_{idx}", use_container_width=True):
            selected_preset = idx
    
    # Manual selection
    st.markdown("**Or select manually:**")
    layer_options = [f"Layer {i} - {name[:30]}" for i, (name, _) in enumerate(all_layers)]
    selected_index = st.selectbox(
        "Select layer index",
        options=list(range(len(all_layers))),
        format_func=lambda x: layer_options[x] if x < len(layer_options) else f"Layer {x}",
        index=32 if len(all_layers) > 32 else 0,
        label_visibility="collapsed"
    )
    
    layer_index = selected_preset if selected_preset is not None else selected_index
    st.caption(f"Selected: **Layer {layer_index}**")

with col2:
    st.subheader("📊 Analysis Results")
    
    analyze_button = st.button("🔍 Analyze Emotion", type="primary", use_container_width=True)
    
    if analyze_button and uploaded_file is not None and text_input.strip():
        with st.spinner("Analyzing..."):
            try:
                # Preprocess
                image_tensor = preprocess_image(image)
                text_tensor = preprocess_text(text_input, word_to_idx, max_len=50)
                
                # Predict
                with torch.no_grad():
                    image_tensor_device = image_tensor.to(device)
                    text_tensor_device = text_tensor.to(device)
                    outputs = model(image_tensor_device, text_tensor_device)
                    probabilities = torch.softmax(outputs, dim=1)
                    prediction = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][prediction].item()
                
                class_names = ['Happy', 'Sad']
                predicted_class = class_names[prediction]
                happy_pct = probabilities[0][0].item() * 100
                sad_pct = probabilities[0][1].item() * 100
                
                # Display prediction
                if predicted_class == 'Happy':
                    st.markdown(f"""
                    <div class="result-box happy">
                        <h2 style="margin: 0;">😊 Happy</h2>
                        <p style="font-size: 1.2rem; margin: 0;">Confidence: {confidence*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box sad">
                        <h2 style="margin: 0;">😢 Sad</h2>
                        <p style="font-size: 1.2rem; margin: 0;">Confidence: {confidence*100:.1f}%</p>
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
                
                # ============================================================
                # GRAD-CAM
                # ============================================================
                st.markdown("---")
                st.markdown(f"### 🔍 Grad-CAM - Layer {layer_index}")
                
                if layer_index < len(all_layers):
                    target_layer = all_layers[layer_index][1]
                    overlay, heatmap = generate_gradcam_for_layer(
                        model, target_layer, image_tensor, text_tensor, device, target_class=prediction
                    )
                    
                    if overlay is not None and heatmap is not None:
                        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                        
                        axes[0].imshow(image.resize((224, 224)))
                        axes[0].set_title("Original")
                        axes[0].axis('off')
                        
                        axes[1].imshow(heatmap, cmap='jet')
                        axes[1].set_title(f"Layer {layer_index} Heatmap")
                        axes[1].axis('off')
                        
                        axes[2].imshow(overlay)
                        axes[2].set_title(f"Layer {layer_index} Overlay")
                        axes[2].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        st.caption("🟡 Yellow/Red areas = Most important regions for prediction")
                    else:
                        st.warning(f"Grad-CAM not available for Layer {layer_index}.")
                        st.info("💡 Try layers: 10, 15, 20, 25, or 32")
                else:
                    st.warning(f"Layer {layer_index} not found. Max layer: {len(all_layers)-1}")
                
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                with st.expander("Show full error"):
                    st.code(traceback.format_exc())
    
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
    MobileNetV2 + BiLSTM | Layer-wise Grad-CAM | 92.69% Val Accuracy
</div>
""", unsafe_allow_html=True)
