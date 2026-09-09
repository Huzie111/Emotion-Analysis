"""
Multi-Method Grad-CAM Streamlit App
Model: MobileNetV2 + BiLSTM (92.69% Validation Accuracy)
Tries multiple Grad-CAM methods until one works
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
    page_title="Emotion Analysis - Multi-Method Grad-CAM",
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
st.markdown('<div class="sub-header">MobileNetV2 + BiLSTM | Multi-Method Explainability</div>', unsafe_allow_html=True)

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
        
        return model, vocab, device
        
    except Exception as e:
        st.error(f"❌ Load failed: {e}")
        return None, None, None

# ============================================================================
# EXTRACT ALL CONV LAYERS
# ============================================================================

def get_all_conv_layers(model):
    """Get all Conv2d layers."""
    layers = []
    def traverse(module, prefix=""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                layers.append((full, child))
            traverse(child, full)
    traverse(model.vision)
    return layers

# ============================================================================
# METHOD 1: Vanilla Grad-CAM
# ============================================================================

class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
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
    
    def generate(self, image, text_tensor, target_class=None):
        self.model.zero_grad()
        img = image.clone().to(self.device).requires_grad_(True)
        txt = text_tensor.clone().to(self.device)
        out = self.model(img, txt)
        if target_class is None:
            target_class = torch.argmax(out, dim=1).item()
        self.model.zero_grad()
        loss = out[0, target_class]
        loss.backward()
        if self.gradients is None or self.activations is None:
            return None, None
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_min, cam_max = torch.min(cam), torch.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            return None, None
        return cam.squeeze().detach().cpu().numpy(), target_class

# ============================================================================
# METHOD 2: Guided Backpropagation
# ============================================================================

class GuidedBackprop:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._register_hooks()
    
    def _register_hooks(self):
        def relu_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0),)
        for module in self.model.vision.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_hook)
    
    def generate(self, image, text_tensor, target_class=None):
        self.model.zero_grad()
        img = image.clone().to(self.device).requires_grad_(True)
        txt = text_tensor.clone().to(self.device)
        out = self.model(img, txt)
        if target_class is None:
            target_class = torch.argmax(out, dim=1).item()
        self.model.zero_grad()
        loss = out[0, target_class]
        loss.backward()
        if img.grad is None:
            return None, None
        grad = img.grad.squeeze().cpu().detach().numpy()
        if grad.shape[0] == 3:
            grad = grad.transpose(1, 2, 0)
        grad = np.abs(grad)
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        return grad, target_class

# ============================================================================
# METHOD 3: Integrated Gradients (Simplified)
# ============================================================================

class IntegratedGradients:
    def __init__(self, model, device, steps=50):
        self.model = model
        self.device = device
        self.steps = steps
    
    def generate(self, image, text_tensor, target_class=None):
        self.model.zero_grad()
        img = image.clone().to(self.device)
        txt = text_tensor.clone().to(self.device)
        
        out = self.model(img, txt)
        if target_class is None:
            target_class = torch.argmax(out, dim=1).item()
        
        # Create baseline (black image)
        baseline = torch.zeros_like(img)
        
        # Integrate gradients
        integrated_grad = torch.zeros_like(img)
        
        for i in range(self.steps):
            alpha = i / self.steps
            interpolated = baseline + alpha * (img - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            out = self.model(interpolated, txt)
            loss = out[0, target_class]
            
            self.model.zero_grad()
            loss.backward()
            
            integrated_grad += interpolated.grad / self.steps
        
        # Convert to visualization
        grad_img = integrated_grad.squeeze().cpu().detach().numpy()
        if grad_img.shape[0] == 3:
            grad_img = grad_img.transpose(1, 2, 0)
        grad_img = np.abs(grad_img)
        grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min() + 1e-8)
        
        return grad_img, target_class

# ============================================================================
# VISUALIZATION
# ============================================================================

def resize_array(arr, target_size):
    """Resize array using PIL."""
    arr_uint8 = (arr * 255).astype(np.uint8)
    if len(arr.shape) == 2:
        arr_pil = Image.fromarray(arr_uint8, mode='L')
    else:
        arr_pil = Image.fromarray(arr_uint8)
    resized = arr_pil.resize(target_size, Image.BILINEAR)
    return np.array(resized) / 255.0

def overlay_heatmap(image, heatmap, alpha=0.6):
    if isinstance(image, torch.Tensor):
        img = image.squeeze().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img = np.clip(img * std + mean, 0, 1)
    else:
        img = np.array(image) / 255.0
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
    
    h, w = img.shape[0], img.shape[1]
    heatmap_resized = resize_array(heatmap, (w, h))
    heatmap_rgb = cm.jet(heatmap_resized)[:, :, :3]
    overlay = (1 - alpha) * img + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay, heatmap_resized

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

# Get all conv layers
all_layers = get_all_conv_layers(model)
st.success(f"✅ Model loaded! Found {len(all_layers)} Conv layers")

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
    
    st.subheader("🎯 Select Method")
    method = st.radio(
        "Choose explainability method:",
        ["Grad-CAM (Layer 0)", "Guided Backprop", "Integrated Gradients", "Try All"],
        index=0,
        horizontal=True
    )

with col2:
    st.subheader("📊 Results")
    analyze = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze and uploaded_file is not None and text_input.strip():
        with st.spinner("Analyzing..."):
            try:
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
                # EXPLANATIONS
                # ============================================================
                st.markdown("---")
                st.markdown("### 🔍 Explainability")
                
                # Find Layer 0
                layer0 = None
                for name, layer in all_layers:
                    if '0' in name and name.count('.') == 1:
                        layer0 = layer
                        break
                if layer0 is None:
                    layer0 = all_layers[0][1]
                
                # Try methods based on selection
                methods_to_try = []
                if method == "Grad-CAM (Layer 0)":
                    methods_to_try = [("Grad-CAM", GradCAM(model, layer0, device))]
                elif method == "Guided Backprop":
                    methods_to_try = [("Guided Backprop", GuidedBackprop(model, device))]
                elif method == "Integrated Gradients":
                    methods_to_try = [("Integrated Gradients", IntegratedGradients(model, device, steps=30))]
                else:  # Try All
                    methods_to_try = [
                        ("Grad-CAM", GradCAM(model, layer0, device)),
                        ("Guided Backprop", GuidedBackprop(model, device)),
                        ("Integrated Gradients", IntegratedGradients(model, device, steps=30))
                    ]
                
                # Try each method
                success_count = 0
                for method_name, method_obj in methods_to_try:
                    try:
                        if isinstance(method_obj, GradCAM):
                            heatmap, target = method_obj.generate(img_tensor, txt_tensor, target_class=pred)
                        else:
                            heatmap, target = method_obj.generate(img_tensor, txt_tensor, target_class=pred)
                        
                        if heatmap is not None:
                            # Enhance heatmap
                            if method_name == "Grad-CAM":
                                heatmap = np.power(heatmap, 0.5)
                            
                            overlay, hm = overlay_heatmap(image, heatmap, alpha=0.6)
                            
                            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                            
                            axes[0].imshow(hm, cmap='jet')
                            axes[0].set_title(f"{method_name}\nHeatmap")
                            axes[0].axis('off')
                            
                            axes[1].imshow(overlay)
                            axes[1].set_title(f"{method_name}\nOverlay")
                            axes[1].axis('off')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            st.caption(f"✅ {method_name} - Max: {np.max(heatmap):.3f} | Mean: {np.mean(heatmap):.3f}")
                            success_count += 1
                            
                    except Exception as e:
                        st.warning(f"⚠️ {method_name} failed: {e}")
                
                if success_count == 0:
                    st.error("❌ All explainability methods failed. Try a different image or text.")
                
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
st.caption("MobileNetV2 + BiLSTM | Multi-Method Explainability | 92.69% Val Accuracy")
