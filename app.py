"""
L0 Grad-CAM Streamlit App
Model: MobileNetV2 + BiLSTM (92.69% Validation Accuracy)
FIXED: Using Layer 0 (features.0.0) - the only compatible layer
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
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
    page_title="Emotion Analysis - L0 Grad-CAM",
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
st.markdown('<div class="sub-header">MobileNetV2 + BiLSTM | L0 Grad-CAM Explainability</div>', unsafe_allow_html=True)

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
        # Load checkpoint to get vocabulary size
        ckpt = torch.load(MODEL_FILE_NAME, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        vocab_size_from_model = state['text_encoder.embedding.weight'].shape[0]
        
        # Load vocabulary and adjust
        vocab_data = torch.load(VOCAB_FILE_NAME, map_location='cpu')
        vocab = vocab_data if isinstance(vocab_data, dict) else {'<PAD>': 0, '<UNK>': 1}
        
        # Ensure vocabulary size matches model
        if len(vocab) != vocab_size_from_model:
            if len(vocab) < vocab_size_from_model:
                for i in range(len(vocab), vocab_size_from_model):
                    vocab[f'_pad_{i}'] = i
            else:
                vocab = dict(list(vocab.items())[:vocab_size_from_model])
        
        vocab_size = len(vocab)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
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
# L0 GRAD-CAM (features.0.0 - The ONLY compatible layer)
# ============================================================================

def get_layer0(model):
    """Get Layer 0 (features.0.0) - the first Conv2d layer."""
    for name, child in model.vision.named_children():
        if name == '0':
            return child
    # Fallback: find first Conv2d
    for module in model.vision.modules():
        if isinstance(module, nn.Conv2d):
            return module
    return None

class L0GradCAM:
    """Grad-CAM using Layer 0 (features.0.0)."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.target_layer = get_layer0(model)
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def fwd(module, inp, out):
            self.activations = out
        def bwd(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(fwd)
        self.target_layer.register_backward_hook(bwd)
    
    def generate_heatmap(self, image, text_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        self.gradients = None
        self.activations = None
        
        # Forward pass
        img = image.clone().to(self.device).requires_grad_(True)
        txt = text_tensor.clone().to(self.device)
        
        output = self.model(img, txt)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # Backward
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        if self.gradients is None or self.activations is None:
            return None
        
        if torch.max(torch.abs(self.gradients)) < 1e-8:
            return None
        
        # Generate heatmap
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam_min, cam_max = torch.min(cam), torch.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        return cam.squeeze().detach().cpu().numpy(), target_class

def overlay_heatmap(image, heatmap, alpha=0.6):
    if isinstance(image, torch.Tensor):
        img = image.squeeze().detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = np.clip(img * std + mean, 0, 1)
    else:
        img = np.array(image) / 255.0
    
    h, w = img.shape[0], img.shape[1]
    heatmap = np.array(Image.fromarray(heatmap).resize((w, h)))
    heatmap = np.clip(heatmap, 0, 1)
    
    heatmap_rgb = cm.jet(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    
    return overlay, heatmap

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

# Verify Layer 0 exists
layer0 = get_layer0(model)
if layer0 is None:
    st.error("❌ Layer 0 not found!")
    st.stop()

st.success("✅ Layer 0 (features.0.0) found for Grad-CAM")

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
                # L0 GRAD-CAM
                # ============================================================
                st.markdown("---")
                st.markdown("### 🔍 L0 Grad-CAM Explanation")
                st.caption("Layer 0 (features.0.0) - Detects basic edges, colors, and simple patterns")
                
                gradcam = L0GradCAM(model, device)
                heatmap, target = gradcam.generate_heatmap(img_tensor, txt_tensor, target_class=pred)
                
                if heatmap is not None:
                    overlay, hm = overlay_heatmap(image, heatmap, alpha=0.6)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                    
                    axes[0].imshow(image.resize((224, 224)))
                    axes[0].set_title("Original Drawing")
                    axes[0].axis('off')
                    
                    axes[1].imshow(hm, cmap='jet')
                    axes[1].set_title("L0 Heatmap")
                    axes[1].axis('off')
                    
                    axes[2].imshow(overlay)
                    axes[2].set_title("L0 Overlay")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.caption("🟡 Yellow/Red areas = Basic features (edges, colors, simple patterns) that influenced the prediction")
                    
                    # Stats
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Max Activation", f"{np.max(heatmap):.3f}")
                    with col_b:
                        st.metric("Mean Activation", f"{np.mean(heatmap):.3f}")
                    with col_c:
                        st.metric("Std Deviation", f"{np.std(heatmap):.3f}")
                    
                    st.info("ℹ️ **Layer 0** captures low-level features like edges, colors, and simple patterns. Since MobileNetV2 uses depthwise separable convolutions, only the first layer is compatible with Grad-CAM.")
                    
                else:
                    st.warning("⚠️ L0 Grad-CAM could not be generated.")
                    st.info("💡 Try a different image or text input.")
                
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
st.caption("MobileNetV2 + BiLSTM | L0 Grad-CAM | 92.69% Val Accuracy")
