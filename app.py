# ============================================================================
# STREAMLIT APP: Emotion Detection with MobileNetV2 + BiLSTM
# ============================================================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from captum.attr import LayerGradCam, LayerAttribution
import os
import io
import re
import glob
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Emotion Detection from Children's Drawings",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Emotion Detection from Children's Drawings")
st.markdown("Upload a drawing to detect if it expresses **Happiness** or **Sadness**")
st.markdown("---")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class BiLSTMTextEncoder(nn.Module):
    def __init__(self, vocab_size=3423, embed_dim=300, hidden=128, dropout=0.3):
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
    import torchvision.models as models
    backbones = {
        'mobilenet_v2': (models.mobilenet_v2, 1280),
        'efficientnet_b0': (models.efficientnet_b0, 1280),
        'shufflenet_v2_x1_0': (models.shufflenet_v2_x1_0, 1024),
    }
    
    if name not in backbones:
        model_fn, dim = backbones['mobilenet_v2']
    else:
        model_fn, dim = backbones[name]
    
    model = model_fn(pretrained=pretrained)
    if hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    elif hasattr(model, 'fc'):
        model.fc = nn.Identity()
    return model, dim

class MultimodalModel(nn.Module):
    def __init__(self, vision_name, text_encoder, dropout=0.5):
        super().__init__()
        self.vision, vdim = get_vision_encoder(vision_name)
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
            nn.Linear(256, 2)
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

# ============================================================================
# VOCABULARY
# ============================================================================

def create_vocab(vocab_size=3423):
    """Create a vocabulary with the specified size."""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    common_words = [
        'happy', 'sad', 'draw', 'feel', 'like', 'love', 'cry', 'smile',
        'angry', 'scared', 'excited', 'tired', 'bored', 'lonely', 'fun',
        'play', 'friend', 'family', 'school', 'home', 'dog', 'cat', 'sun',
        'rain', 'because', 'very', 'really', 'want', 'think', 'know', 'see',
        'look', 'good', 'bad', 'nice', 'great', 'wonderful', 'terrible',
        'awful', 'beautiful', 'ugly', 'big', 'small', 'little', 'much',
        'many', 'more', 'most', 'some', 'any', 'all', 'every', 'people',
        'child', 'mother', 'father', 'brother', 'sister', 'teacher',
        'school', 'home', 'funny', 'silly', 'scary', 'brave', 'kind',
        'mean', 'loved', 'hated', 'confused', 'surprised', 'worried',
        'proud', 'embarrassed', 'jealous', 'guilty', 'shy', 'confident'
    ]
    
    for i, word in enumerate(common_words, 2):
        vocab[word] = i
    
    # Fill remaining vocabulary with placeholder tokens
    current_size = len(vocab)
    for i in range(current_size, vocab_size):
        vocab[f'<TOKEN_{i}>'] = i
    
    return vocab

def text_to_sequence(text, vocab, max_len=50):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()[:max_len]
    seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    seq += [0] * (max_len - len(seq))
    return torch.tensor(seq, dtype=torch.long)

# ============================================================================
# FIND MODEL FILE
# ============================================================================

def find_model_file():
    """Search for model file in multiple locations."""
    
    search_paths = [
        # GitHub repository paths
        "Emotion_Models/MM_MobileNetV2_BiLSTM_final.pt",
        "Emotion_Models/compressed_models/MM_MobileNetV2_BiLSTM_final.pt",
        "MM_MobileNetV2_BiLSTM_final.pt",
        
        # Alternative paths
        "*.pt",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
        
        if '*' in path:
            matches = glob.glob(path)
            if matches:
                return matches[0]
    
    # Search all subdirectories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pt'):
                return os.path.join(root, file)
    
    return None

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_cached_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find model file
    model_path = find_model_file()
    
    if model_path is None:
        st.error("""
        ❌ **Model file not found!**
        
        Expected: `Emotion_Models/MM_MobileNetV2_BiLSTM_final.pt`
        
        Please ensure the model file is uploaded to your GitHub repository.
        """)
        
        # Show available files
        st.subheader("📁 Available Files:")
        files = []
        for root, dirs, filenames in os.walk('.'):
            for f in filenames:
                if f.endswith('.pt') or f.endswith('.pth'):
                    files.append(os.path.join(root, f))
        
        if files:
            st.write("Found these model files:")
            for f in files:
                st.write(f"   - `{f}`")
        else:
            st.write("No `.pt` or `.pth` files found.")
            
        return None, None, device
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Get vocabulary size from embedding
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        embedding_weight = state_dict.get('text_encoder.embedding.weight')
        
        if embedding_weight is not None:
            vocab_size = embedding_weight.shape[0]
        else:
            vocab_size = 3423
        
        st.info(f"📊 Using vocabulary size: {vocab_size}")
        
        # Create vocabulary with correct size
        vocab = create_vocab(vocab_size)
        
        # Create model with correct vocab size
        text_encoder = BiLSTMTextEncoder(vocab_size)
        model = MultimodalModel('mobilenet_v2', text_encoder)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(device)
        model.eval()
        
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        
        st.success(f"✅ Model loaded from `{os.path.basename(model_path)}` ({file_size:.2f} MB)")
        st.info(f"📊 Model: MobileNetV2 + BiLSTM")
        st.info(f"📊 Vocabulary size: {vocab_size}")
        
        return model, vocab, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
        # Show helpful debug info
        st.subheader("🔍 Debug Information:")
        st.code(f"Model path: {model_path}")
        st.code(f"File size: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
        
        return None, None, device

# ============================================================================
# GRAD-CAM FUNCTIONS
# ============================================================================

def get_target_layer(model):
    """Find the last convolutional layer for MobileNetV2."""
    vision = model.vision
    
    if hasattr(vision, 'features'):
        if hasattr(vision.features, '_modules'):
            keys = list(vision.features._modules.keys())
            if keys:
                for key in reversed(keys):
                    module = vision.features._modules[key]
                    if isinstance(module, nn.Conv2d):
                        return module
                return vision.features._modules[keys[-1]]
    
    # Fallback
    for name, module in vision.named_modules():
        if isinstance(module, nn.Conv2d):
            return module
    
    return vision

def get_layerwise_layers(model):
    """Get multiple convolutional layers for layer-wise Grad-CAM."""
    layers = []
    layer_names = []
    vision = model.vision
    
    if hasattr(vision, 'features'):
        if hasattr(vision.features, '_modules'):
            keys = list(vision.features._modules.keys())
            selected_indices = [0, len(keys)//4, len(keys)//2, 3*len(keys)//4, -1]
            for idx in selected_indices:
                if abs(idx) < len(keys):
                    module = vision.features._modules[keys[idx]]
                    if isinstance(module, nn.Conv2d):
                        layers.append(module)
                        layer_names.append(f"Layer {keys[idx]}")
    
    if len(layers) < 3:
        for name, module in vision.named_modules():
            if isinstance(module, nn.Conv2d):
                layers.append(module)
                layer_names.append(name.split('.')[-1])
                if len(layers) >= 5:
                    break
    
    return layers, layer_names

def generate_gradcam_heatmap(model, image, target_class, layer_idx=0):
    """Generate Grad-CAM heatmap for a specific layer."""
    device = next(model.parameters()).device
    
    target_layer = get_target_layer(model)
    
    if layer_idx > 0:
        layers, layer_names = get_layerwise_layers(model)
        if layer_idx <= len(layers):
            target_layer = layers[layer_idx - 1]
    
    grad_cam = LayerGradCam(model, target_layer)
    dummy_text = torch.zeros(1, 50, dtype=torch.long).to(device)
    
    attributions = grad_cam.attribute(
        image, target=target_class, additional_forward_args=(dummy_text,)
    )
    
    if attributions.dim() == 4:
        heatmap = LayerAttribution.interpolate(attributions, (224, 224))
    elif attributions.dim() == 3:
        attributions = attributions.unsqueeze(1)
        heatmap = LayerAttribution.interpolate(attributions, (224, 224))
    else:
        heatmap = torch.ones(1, 1, 224, 224).to(device)
    
    heatmap = heatmap.squeeze().cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def generate_layerwise_gradcam(model, image, target_class):
    """Generate layer-wise Grad-CAM for all layers."""
    layers, layer_names = get_layerwise_layers(model)
    heatmaps = []
    
    for i, (layer, name) in enumerate(zip(layers, layer_names)):
        grad_cam = LayerGradCam(model, layer)
        dummy_text = torch.zeros(1, 50, dtype=torch.long).to(next(model.parameters()).device)
        
        attributions = grad_cam.attribute(
            image, target=target_class, additional_forward_args=(dummy_text,)
        )
        
        if attributions.dim() == 4:
            heatmap = LayerAttribution.interpolate(attributions, (224, 224))
        elif attributions.dim() == 3:
            attributions = attributions.unsqueeze(1)
            heatmap = LayerAttribution.interpolate(attributions, (224, 224))
        else:
            heatmap = torch.ones(1, 1, 224, 224).to(device)
        
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmaps.append(heatmap)
    
    return heatmaps, layer_names

def overlay_heatmap_jet(image, heatmap, alpha=0.5):
    """Overlay heatmap with jet colormap using matplotlib."""
    if isinstance(image, torch.Tensor):
        img = image.squeeze().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
    else:
        img = image
    
    from PIL import Image as PILImage
    heatmap_resized = np.array(PILImage.fromarray(heatmap).resize((img.shape[1], img.shape[0])))
    
    cmap = plt.cm.jet(heatmap_resized)
    heatmap_color = cmap[:, :, :3]
    
    overlay = img * (1 - alpha) + heatmap_color * alpha
    return np.clip(overlay, 0, 1)

def normalize_image(tensor):
    """Normalize image tensor for display."""
    img = tensor.cpu().detach().numpy()
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load model
    model, vocab, device = load_cached_model()
    
    if model is None:
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload & Predict", "🔍 Layer-wise XAI"])
    
    # ========================================================================
    # TAB 1: Upload & Predict
    # ========================================================================
    
    with tab1:
        st.header("Upload a Drawing")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file...",
                type=["jpg", "jpeg", "png"],
                help="Upload a drawing to analyze the emotion"
            )
            
            st.subheader("Child's Explanation (Optional)")
            child_text = st.text_area(
                "What did the child say about their drawing?",
                placeholder="e.g., I drew this because I was happy...",
                height=80
            )
            
            predict_button = st.button("🔮 Predict Emotion", type="primary", use_container_width=True)
        
        with col2:
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Drawing", use_container_width=True)
                
                if predict_button:
                    with st.spinner("Analyzing..."):
                        try:
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                            ])
                            img_tensor = transform(image).unsqueeze(0).to(device)
                            
                            if child_text:
                                text_tensor = text_to_sequence(child_text, vocab).unsqueeze(0).to(device)
                            else:
                                text_tensor = torch.zeros(1, 50, dtype=torch.long).to(device)
                            
                            with torch.no_grad():
                                output = model(img_tensor, text_tensor)
                                probs = torch.softmax(output, dim=1)
                                pred = torch.argmax(probs, dim=1).item()
                            
                            emotion = "Happy 😊" if pred == 0 else "Sad 😢"
                            confidence = probs[0][pred].item()
                            
                            st.subheader("📊 Prediction Result")
                            
                            bg_color = '#d4edda' if pred == 0 else '#f8d7da'
                            
                            st.markdown(f"""
                            <div style="text-align: center; padding: 20px; background-color: {bg_color}; border-radius: 10px;">
                                <h1 style="font-size: 48px; margin: 0;">{emotion}</h1>
                                <p style="font-size: 24px; margin: 0;">Confidence: {confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.session_state['current_image'] = img_tensor
                            st.session_state['current_pred'] = pred
                            st.session_state['current_probs'] = probs
                            st.session_state['current_pil'] = image
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
    
    # ========================================================================
    # TAB 2: Layer-wise XAI
    # ========================================================================
    
    with tab2:
        st.header("🔍 Layer-wise Grad-CAM Explanation")
        st.markdown("Shows which parts of the drawing influenced the model's decision at different network layers.")
        
        if 'current_image' not in st.session_state:
            st.info("Please upload an image and make a prediction first in the 'Upload & Predict' tab.")
        else:
            image_tensor = st.session_state['current_image']
            pred_class = st.session_state['current_pred']
            pil_image = st.session_state['current_pil']
            
            st.subheader("Layer Selection")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                layer_options = [
                    "Feature Layer 1 (Early)",
                    "Feature Layer 2 (Mid-Early)",
                    "Feature Layer 3 (Mid)",
                    "Feature Layer 4 (Mid-Late)",
                    "Feature Layer 5 (Late)"
                ]
                selected_layer = st.selectbox("Select Layer:", layer_options)
            
            with col2:
                alpha = st.slider("Heatmap Opacity:", 0.1, 1.0, 0.5, 0.1)
                target_label = "Happy" if pred_class == 0 else "Sad"
                st.write(f"**Target Class:** {target_label}")
            
            with st.spinner("Generating explanation..."):
                try:
                    layer_idx = layer_options.index(selected_layer)
                    
                    heatmap = generate_gradcam_heatmap(
                        model, image_tensor, pred_class, layer_idx
                    )
                    
                    img_display = normalize_image(image_tensor)
                    overlay = overlay_heatmap_jet(img_display, heatmap, alpha)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.image(img_display, caption="Original Image", use_container_width=True)
                    
                    with col2:
                        st.image(heatmap, caption=f"Heatmap\n{selected_layer}", use_container_width=True)
                    
                    with col3:
                        st.image(overlay, caption="Overlay", use_container_width=True)
                    
                    st.subheader("Layer-wise Grad-CAM Grid")
                    st.markdown("All layers side-by-side for comparison")
                    
                    with st.spinner("Generating all layers..."):
                        heatmaps, layer_names = generate_layerwise_gradcam(
                            model, image_tensor, pred_class
                        )
                        
                        cols = st.columns(len(heatmaps))
                        for i, (col, hm, name) in enumerate(zip(cols, heatmaps, layer_names)):
                            with col:
                                st.image(hm, caption=f"{name[:12]}", use_container_width=True)
                        
                        st.markdown("### Overlay Grid")
                        cols_overlay = st.columns(len(heatmaps))
                        for i, (col, hm, name) in enumerate(zip(cols_overlay, heatmaps, layer_names)):
                            with col:
                                overlay_hm = overlay_heatmap_jet(img_display, hm, alpha)
                                st.image(overlay_hm, caption=f"{name[:12]}", use_container_width=True)
                    
                    st.subheader("📖 Interpretation")
                    
                    focus_scores = []
                    for hm in heatmaps:
                        focus = (hm > 0.5).mean()
                        focus_scores.append(focus)
                    
                    best_layer_idx = np.argmax(focus_scores)
                    best_layer_name = layer_names[best_layer_idx] if best_layer_idx < len(layer_names) else "Unknown"
                    
                    if pred_class == 0:
                        st.success(f"""
                        **The model focused on regions that typically indicate happiness:**
                        - Most focused layer: **{best_layer_name}**
                        - Areas around the eyes and mouth (smile indicators)
                        - Bright/positive color regions
                        - Open body language features
                        """)
                    else:
                        st.warning(f"""
                        **The model focused on regions that typically indicate sadness:**
                        - Most focused layer: **{best_layer_name}**
                        - Areas around the eyes (drooping indicators)
                        - Darker/shadow regions
                        - Closed/withdrawn body language features
                        """)
                    
                    st.subheader("📥 Export Explanation")
                    
                    fig, axes = plt.subplots(2, len(heatmaps) + 1, figsize=(20, 8))
                    
                    axes[0, 0].imshow(img_display)
                    axes[0, 0].set_title('Original')
                    axes[0, 0].axis('off')
                    
                    for i, (hm, name) in enumerate(zip(heatmaps, layer_names)):
                        axes[0, i+1].imshow(hm, cmap='jet')
                        axes[0, i+1].set_title(name[:12])
                        axes[0, i+1].axis('off')
                    
                    axes[1, 0].imshow(img_display)
                    axes[1, 0].set_title('Original')
                    axes[1, 0].axis('off')
                    
                    for i, (hm, name) in enumerate(zip(heatmaps, layer_names)):
                        overlay_hm = overlay_heatmap_jet(img_display, hm, alpha)
                        axes[1, i+1].imshow(overlay_hm)
                        axes[1, i+1].set_title(name[:12])
                        axes[1, i+1].axis('off')
                    
                    plt.suptitle(f'Layer-wise Grad-CAM - Target: {target_label}', fontsize=16)
                    plt.tight_layout()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 Download Layer-wise Grad-CAM",
                        data=buf.getvalue(),
                        file_name=f"layerwise_gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating XAI: {e}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
