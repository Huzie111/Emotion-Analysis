"""
Layer-wise Grad-CAM Streamlit App
Model: MobileNetV2 + BiLSTM (92.69% Validation Accuracy)
Using PROVEN Grad-CAM approach from your working notebook
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import cv2
import gdown
import requests
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

st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none !important; }
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎨 Emotion Analysis from Children\'s Drawings</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MobileNetV2 + BiLSTM | Layer-wise Grad-CAM Explainability</div>', unsafe_allow_html=True)

# ============================================================================
# GOOGLE DRIVE DOWNLOAD
# ============================================================================

MODEL_FILE_ID = "11lYY2-0tXlF4mE1peB2ReQy9bMLlp2mp"
VOCAB_FILE_ID = "1r2mCVi-tVjeI18P2dBFFdlYAeHNuKnm-"
MODEL_FILE_NAME = "MM_MobileNetV2_BiLSTM_final.pt"
VOCAB_FILE_NAME = "vocabulary.pth"

def download_file(file_id, output_path, desc="file"):
    try:
        st.info(f"📥 Downloading {desc}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.success(f"✅ {desc} downloaded!")
            return True
    except:
        try:
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            r = requests.get(url, stream=True)
            if 'confirm' in r.text:
                confirm = re.search(r'confirm=([^&]+)', r.text).group(1)
                r = requests.get(f"{url}&confirm={confirm}", stream=True)
            if r.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        if chunk: f.write(chunk)
                return True
        except: pass
    return False

def check_files():
    if not os.path.exists(MODEL_FILE_NAME):
        if not download_file(MODEL_FILE_ID, MODEL_FILE_NAME, "model"): return False
    if not os.path.exists(VOCAB_FILE_NAME):
        if not download_file(VOCAB_FILE_ID, VOCAB_FILE_NAME, "vocabulary"): return False
    return True

# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden=128, dropout=0.7):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, 2, bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden * 2
    def forward(self, x):
        x = self.dropout(self.embedding(x))
        x, _ = self.lstm(x)
        attn = torch.softmax(self.attention(x), dim=1)
        return (attn * x).sum(dim=1)

def get_vision(name, pretrained=False):
    backbones = {'mobilenet_v2': (models.mobilenet_v2, 1280)}
    if name not in backbones: raise ValueError(f"Unknown: {name}")
    model, dim = backbones[name]
    model = model(pretrained=pretrained)
    if hasattr(model, 'classifier'): model.classifier = nn.Identity()
    return model, dim

class MultimodalModel(nn.Module):
    def __init__(self, vision_name, text_encoder, dropout=0.7):
        super().__init__()
        self.vision, vdim = get_vision(vision_name, pretrained=False)
        self.vision_proj = nn.Sequential(nn.Linear(vdim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout))
        self.text_encoder = text_encoder
        self.text_proj = nn.Sequential(nn.Linear(text_encoder.output_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    def forward(self, images, texts):
        v = self.vision(images)
        if v.dim() > 2: v = v.mean([2, 3])
        v = self.vision_proj(v)
        t = self.text_encoder(texts)
        t = self.text_proj(t)
        return self.classifier(torch.cat([v, t], dim=1))

def create_text_encoder(text_type, vocab_size, hidden=128):
    if text_type == 'bilstm': return BiLSTMTextEncoder(vocab_size, hidden=hidden)
    raise ValueError(f"Unknown: {text_type}")

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    if not check_files(): return None, None, None
    
    try:
        ckpt = torch.load(MODEL_FILE_NAME, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        vocab_size_from_model = state['text_encoder.embedding.weight'].shape[0]
        
        vocab_data = torch.load(VOCAB_FILE_NAME, map_location='cpu')
        vocab = vocab_data if isinstance(vocab_data, dict) else {'<PAD>': 0, '<UNK>': 1}
        
        if len(vocab) != vocab_size_from_model:
            if len(vocab) < vocab_size_from_model:
                for i in range(len(vocab), vocab_size_from_model):
                    vocab[f'_pad_{i}'] = i
            else:
                vocab = dict(list(vocab.items())[:vocab_size_from_model])
        
        vocab_size = len(vocab)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        text_enc = create_text_encoder('bilstm', vocab_size, hidden=128)
        model = MultimodalModel('mobilenet_v2', text_enc)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        
        st.success("✅ Model loaded successfully!")
        return model, vocab, device
        
    except Exception as e:
        st.error(f"❌ Load failed: {e}")
        return None, None, None

# ============================================================================
# EXTRACT ALL CONVOLUTIONAL LAYERS (YOUR PROVEN APPROACH)
# ============================================================================

def get_all_conv_layers(model, max_layers=50):
    """Extract all convolutional layers from vision encoder."""
    layers = {}
    layer_count = 0
    prefix = model.vision
    
    def traverse(module, path=""):
        nonlocal layer_count
        for name, child in module.named_children():
            new_path = f"{path}.{name}" if path else name
            if isinstance(child, nn.Conv2d):
                layer_name = f"Conv_{layer_count}"
                layers[layer_name] = child
                layer_count += 1
                if layer_count >= max_layers:
                    return
            traverse(child, new_path)
    
    traverse(prefix)
    return layers

# ============================================================================
# LAYER-WISE GRAD-CAM (YOUR PROVEN APPROACH)
# ============================================================================

class LayerWiseGradCAM:
    """Grad-CAM across multiple layers (YOUR PROVEN APPROACH)."""
    
    def __init__(self, model, target_layers, device):
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.gradients = {}
        self.activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        for name, layer in self.target_layers.items():
            def forward_hook(module, input, output, name=name):
                self.activations[name] = output
                
            def backward_hook(module, grad_input, grad_output, name=name):
                self.gradients[name] = grad_output[0]
            
            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)
    
    def generate_layer_heatmaps(self, image, text_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        # Prepare inputs
        img = image.clone().to(self.device).requires_grad_(True)
        txt = text_tensor.clone().to(self.device)
        
        # Forward pass
        output = self.model(img, txt)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Generate heatmaps for each layer
        heatmaps = {}
        
        for name in self.activations.keys():
            activations = self.activations[name]
            gradients = self.gradients[name]
            
            # Global average pooling of gradients
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            
            # Weighted combination
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            
            heatmaps[name] = cam.squeeze().cpu().detach().numpy()
        
        return heatmaps, target_class

# ============================================================================
# VISUALIZATION (YOUR PROVEN APPROACH)
# ============================================================================

def visualize_layer_heatmaps(original_image, heatmaps, layer_names, sample_label, sample_id, target_class):
    """Visualize heatmaps for all layers (YOUR PROVEN APPROACH)."""
    
    # Convert original image
    if isinstance(original_image, torch.Tensor):
        img = original_image.squeeze().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = np.clip(img * std + mean, 0, 1)
    else:
        img = np.array(original_image) / 255.0
    
    n_layers = len(heatmaps)
    n_cols = min(4, n_layers + 1)
    n_rows = (n_layers + 1 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f'Original\n{sample_id}\nLabel: {sample_label}')
    axes[0].axis('off')
    
    # Plot each layer
    for idx, (name, heatmap) in enumerate(heatmaps.items(), 1):
        if idx >= len(axes):
            break
        
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        axes[idx].imshow(img)
        axes[idx].imshow(heatmap_resized, cmap='jet', alpha=0.5)
        layer_short = name.replace('Conv_', 'L')
        axes[idx].set_title(f'{layer_short}\nLayer {idx}')
        axes[idx].axis('off')
    
    # Hide unused axes
    for i in range(len(heatmaps) + 1, len(axes)):
        axes[i].axis('off')
    
    class_label = 'Happy' if target_class == 0 else 'Sad'
    plt.suptitle(f'Layer-wise Grad-CAM: Predicted = {target_class} ({class_label})', fontsize=14)
    plt.tight_layout()
    return fig

# ============================================================================
# LIME FOR TEXT EXPLANATION
# ============================================================================

def generate_lime_text_explanation(text, model, word_to_idx, device, max_len=50):
    """Generate LIME-style text explanation (YOUR PROVEN APPROACH)."""
    
    words = text.lower().split()
    if len(words) == 0:
        return None, []
    
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
# PREPROCESSING
# ============================================================================

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def preprocess_text(text, vocab, max_len=50):
    tokens = text.lower().split()
    ids = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# ============================================================================
# MAIN APP
# ============================================================================

model, vocab, device = load_model()
if model is None:
    st.stop()

# Get all conv layers (YOUR PROVEN APPROACH)
all_conv_layers = get_all_conv_layers(model, max_layers=50)
st.success(f"✅ Found {len(all_conv_layers)} convolutional layers")

# ============================================================================
# UI
# ============================================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Drawing")
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Drawing", use_container_width=True)
    else:
        st.info("Upload a drawing (JPG/PNG)")
        image = None
    
    st.subheader("✍️ Self-Reflection")
    text_input = st.text_area("", placeholder="I drew this because I felt...", height=80, label_visibility="collapsed")
    
    st.subheader("🎯 Select Grad-CAM Layers")
    
    # Layer selection - show all layers
    layer_options = list(all_conv_layers.keys())
    selected_layers = st.multiselect(
        "Select layers to visualize (leave empty for all 8)",
        options=layer_options,
        default=layer_options[:8] if len(layer_options) >= 8 else layer_options,
        label_visibility="collapsed"
    )
    
    if not selected_layers:
        selected_layers = layer_options[:8] if len(layer_options) >= 8 else layer_options

with col2:
    st.subheader("📊 Results")
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze and uploaded_file is not None and text_input.strip():
        with st.spinner("Analyzing..."):
            try:
                # Prepare inputs
                img_tensor = preprocess_image(image)
                txt_tensor = preprocess_text(text_input, vocab)
                
                # Predict
                with torch.no_grad():
                    out = model(img_tensor.to(device), txt_tensor.to(device))
                    probs = torch.softmax(out, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    conf = probs[0, pred].item()
                
                # Display prediction
                if pred == 0:
                    st.markdown(f"""
                    <div class="result-box happy">
                        <h2 style="margin:0;">😊 Happy</h2>
                        <p style="font-size:1.2rem;margin:0;">Confidence: {conf*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box sad">
                        <h2 style="margin:0;">😢 Sad</h2>
                        <p style="font-size:1.2rem;margin:0;">Confidence: {conf*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ============================================================
                # LAYER-WISE GRAD-CAM (YOUR PROVEN APPROACH)
                # ============================================================
                st.markdown("---")
                st.markdown("### 🔍 Layer-wise Grad-CAM Explanation")
                
                # Select target layers from all_conv_layers
                target_layers = {}
                for name in selected_layers:
                    if name in all_conv_layers:
                        target_layers[name] = all_conv_layers[name]
                
                # Initialize Layer-Wise Grad-CAM
                layer_cam = LayerWiseGradCAM(model, target_layers, device)
                
                try:
                    # Generate heatmaps
                    heatmaps, target_class = layer_cam.generate_layer_heatmaps(
                        img_tensor, txt_tensor, target_class=pred
                    )
                    
                    if heatmaps and len(heatmaps) > 0:
                        # Visualize
                        fig = visualize_layer_heatmaps(
                            image, heatmaps, list(heatmaps.keys()),
                            "Drawing", uploaded_file.name, target_class
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        st.caption("🟡 Yellow/Red areas = Regions most important for the prediction")
                        
                        # Show activation stats
                        st.markdown("**Layer Activation Statistics:**")
                        stats_cols = st.columns(min(4, len(heatmaps)))
                        for i, (name, heatmap) in enumerate(heatmaps.items()):
                            if i < len(stats_cols):
                                with stats_cols[i]:
                                    st.metric(
                                        name.replace('Conv_', 'L'),
                                        f"max: {np.max(heatmap):.3f}\navg: {np.mean(heatmap):.3f}"
                                    )
                    else:
                        st.warning("⚠️ No heatmaps generated. Try different layers.")
                        
                except Exception as e:
                    st.warning(f"⚠️ Grad-CAM error: {e}")
                    st.info("💡 Try selecting different layers or check your inputs")
                
                # ============================================================
                # LIME - Text Explanation (YOUR PROVEN APPROACH)
                # ============================================================
                st.markdown("---")
                st.markdown("### 📝 LIME: Text Explanation")
                
                try:
                    base_pred, word_importance = generate_lime_text_explanation(
                        text_input, model, vocab, device, max_len=50
                    )
                    
                    if word_importance and len(word_importance) > 0:
                        highlighted_words = []
                        for word, importance in word_importance:
                            if importance > 0.7:
                                cls = "high"
                                color = "#dc3545"
                            elif importance > 0.4:
                                cls = "medium"
                                color = "#ffc107"
                            else:
                                cls = "low"
                                color = "#e9ecef"
                            
                            text_color = "white" if importance > 0.7 else "#333"
                            highlighted_words.append(
                                f'<span style="background: {color}; color: {text_color}; padding: 2px 6px; margin: 1px; border-radius: 4px;">{word}</span>'
                            )
                        
                        st.markdown(
                            f'<div style="padding: 10px; background: #f8f9fa; border-radius: 8px; font-size: 0.95rem; line-height: 1.8;">'
                            f'{" ".join(highlighted_words)}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                        st.caption("🔴 High importance | 🟡 Medium | ⚪ Low")
                        class_names = ['Happy', 'Sad']
                        st.caption(f"Text-based prediction: {class_names[base_pred]}")
                    else:
                        st.info("LIME explanation not available for this text")
                except Exception as e:
                    st.info(f"LIME explanation not available: {e}")
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.code(str(e))
    
    elif analyze:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a drawing")
        if not text_input.strip():
            st.warning("⚠️ Please enter self-reflection text")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("MobileNetV2 + BiLSTM | Layer-wise Grad-CAM | 92.69% Val Accuracy")
