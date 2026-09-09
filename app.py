# ============================================================================
# COMPLETE FIXED NOTEBOOK: Layer-wise Grad-CAM with LIME
# ============================================================================
# This notebook extracts all layers, visualizes Grad-CAM for Layer 0,
# and includes LIME for text explanations
# ============================================================================

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

print("="*60)
print("✅ LAYER-WISE GRAD-CAM WITH LIME")
print("="*60)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = "Emotion_Models_Retrained/"
IMAGES_DIR = "KIDO/Images/Emotion"
TEST_CSV = "KIDO/Texts/Emotion/Emotion_Test.csv"
TRAIN_CSV = "KIDO/Texts/Emotion/Emotion_Train.csv"

BATCH_SIZE = 16
MAX_TEXT_LENGTH = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Model Directory:  {MODEL_DIR}")
print(f"Device:           {DEVICE}")
print("="*60)

# ============================================================================
# LOAD MODEL AND VOCABULARY
# ============================================================================

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Zğüşıöç\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocabulary_from_training(target_size):
    from collections import Counter
    train_df = pd.read_csv(TRAIN_CSV, header=None)
    texts = [str(row[2]) for _, row in train_df.iterrows() if pd.notna(row[2])]
    
    word_counts = Counter()
    for text in texts:
        tokens = preprocess_text(text).split()
        word_counts.update(tokens)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    max_words = target_size - 2
    for word, _ in word_counts.most_common(max_words):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    while len(vocab) < target_size:
        vocab[f'_dummy_{len(vocab)}'] = len(vocab)
    
    return vocab

# Model components
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
    else:
        raise ValueError(f"Unknown text encoder: {text_type}")

def load_model_with_vocab(model_path, model_name, vocab, device):
    checkpoint = torch.load(model_path, map_location=device)
    vocab_size = len(vocab)
    
    parts = model_name.split('_')
    vision_part = parts[1].lower() if len(parts) >= 3 else 'mobilenet_v2'
    text_part = parts[2].lower() if len(parts) >= 3 else 'bilstm'
    
    if 'squeeze' in vision_part:
        vision_name = 'squeezenet1_1'
    elif 'shuffle' in vision_part:
        vision_name = 'shufflenet_v2_x1_0'
    elif 'efficient' in vision_part:
        vision_name = 'efficientnet_b0'
    else:
        vision_name = 'mobilenet_v2'
    
    text_type = 'gru' if 'gru' in text_part else 'bilstm'
    hidden = 64 if 'squeeze' in vision_part else 128
    
    text_enc = create_text_encoder(text_type, vocab_size, hidden)
    model = MultimodalModel(vision_name, text_enc)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

# Find best model
def find_best_model(model_dir):
    best_model = None
    best_acc = -1
    for f in os.listdir(model_dir):
        if f.endswith('_final.pt'):
            path = os.path.join(model_dir, f)
            checkpoint = torch.load(path, map_location='cpu')
            acc = checkpoint.get('best_validation_accuracy', 0)
            if acc > best_acc:
                best_acc = acc
                best_model = f.replace('_final.pt', '')
    return best_model, best_acc

best_model_name, best_acc = find_best_model(MODEL_DIR)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Best Validation Accuracy: {best_acc:.4f}")

# Load model
model_path = os.path.join(MODEL_DIR, f"{best_model_name}_final.pt")
checkpoint = torch.load(model_path, map_location='cpu')
vocab_size = checkpoint['model_state_dict']['text_encoder.embedding.weight'].shape[0]
print(f"✅ Vocabulary size from model: {vocab_size}")

vocab = build_vocabulary_from_training(vocab_size)
print(f"✅ Vocabulary built: {len(vocab)} words")

model, checkpoint = load_model_with_vocab(model_path, best_model_name, vocab, DEVICE)
print(f"✅ Model loaded: {best_model_name}")

# ============================================================================
# LOAD SAMPLE IMAGES
# ============================================================================

def load_test_data():
    if not os.path.exists(TEST_CSV):
        print("❌ Test CSV not found!")
        return None
    
    test_df = pd.read_csv(TEST_CSV, header=None)
    data = []
    
    for _, row in test_df.iterrows():
        try:
            img_id = str(row[0]).strip()
            text = str(row[2]) if pd.notna(row[2]) else ""
            label_raw = str(row[3]).strip().lower()
            
            if label_raw in ['happiness', 'happy']:
                label = 'happy'
                label_dir = 'Happiness'
            elif label_raw in ['sadness', 'sad']:
                label = 'sad'
                label_dir = 'Sadness'
            else:
                continue
            
            img_path = os.path.join(IMAGES_DIR, "test", label_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGES_DIR, "test", label_dir, f"{img_id}.jpg")
                if not os.path.exists(img_path):
                    continue
            
            data.append({
                'image_id': img_id,
                'image_path': img_path,
                'label': label,
                'label_dir': label_dir,
                'text': text
            })
        except:
            continue
    
    return pd.DataFrame(data)

test_df = load_test_data()
if test_df is not None:
    print(f"✅ Test data loaded: {len(test_df)} samples")
    
    # Get samples
    samples = []
    for label in ['happy', 'sad']:
        subset = test_df[test_df['label'] == label].head(2)
        for _, row in subset.iterrows():
            try:
                img = Image.open(row['image_path']).convert('RGB')
                samples.append({
                    'image': img,
                    'path': row['image_path'],
                    'label': row['label'],
                    'image_id': row['image_id'],
                    'text': row['text']
                })
            except:
                continue
    
    print(f"✅ Loaded {len(samples)} sample images")

# ============================================================================
# EXTRACT ALL CONVOLUTIONAL LAYERS
# ============================================================================

def get_all_conv_layers(model, max_layers=None):
    """Extract all convolutional layers from vision encoder."""
    
    layers = {}
    layer_count = 0
    prefix = model.vision
    
    def traverse(module, path=""):
        nonlocal layer_count
        
        for name, child in module.named_children():
            new_path = f"{path}.{name}" if path else name
            
            # Check if it's a convolutional layer
            if isinstance(child, nn.Conv2d):
                # Check if compatible with Grad-CAM
                k = child.kernel_size
                if isinstance(k, tuple):
                    is_compatible = (k[0] > 1 or k[1] > 1) and child.groups == 1
                else:
                    is_compatible = k > 1 and child.groups == 1
                
                layer_name = f"Conv_{layer_count}"
                layers[layer_name] = {
                    'layer': child,
                    'compatible': is_compatible,
                    'kernel_size': k,
                    'groups': child.groups,
                    'in_channels': child.in_channels,
                    'out_channels': child.out_channels
                }
                layer_count += 1
                if max_layers and layer_count >= max_layers:
                    return
            
            # Traverse deeper
            traverse(child, new_path)
    
    traverse(prefix)
    return layers

# Get conv layers
all_conv_layers = get_all_conv_layers(model, max_layers=None)
print(f"\n✅ Found {len(all_conv_layers)} convolutional layers")

# Show layers
print("\n📋 LAYERS FOUND:")
for i, (name, info) in enumerate(all_conv_layers.items()):
    compatible = "✅" if info['compatible'] else "❌"
    print(f"   {compatible} {name}: kernel={info['kernel_size']}, groups={info['groups']}, compatible={info['compatible']}")

# ============================================================================
# LAYER-WISE GRAD-CAM (Using Layer 0 - The ONLY compatible layer)
# ============================================================================

class LayerWiseGradCAM:
    """Grad-CAM for Layer 0 (features.0.0)."""
    
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
    
    def generate_layer_heatmaps(self, image, text):
        """Generate heatmap for Layer 0."""
        
        self.model.eval()
        self.model.zero_grad()
        
        # Prepare inputs
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        if isinstance(image, Image.Image):
            img = transform(image).unsqueeze(0).to(self.device).requires_grad_(True)
        else:
            img = image.unsqueeze(0).to(self.device).requires_grad_(True)
        
        # Prepare text
        tokens = preprocess_text(text).split()[:50]
        seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        seq += [0] * (50 - len(seq))
        txt = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Forward pass
        output = self.model(img, txt)
        target_class = torch.argmax(output, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Generate heatmap
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            return None, None, None
        
        if torch.max(torch.abs(gradients)) < 1e-8:
            return None, None, None
        
        # Global average pooling
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam_min = torch.min(cam)
        cam_max = torch.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            return None, None, None
        
        heatmap = cam.squeeze().detach().cpu().numpy()
        
        return heatmap, target_class, loss.item()

def visualize_heatmap(image, heatmap, title="Grad-CAM"):
    """Visualize heatmap overlay."""
    
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        img = image.squeeze().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.array(image) / 255.0
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Heatmap
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title("Heatmap")
    axes[1].axis('off')
    
    # Overlay
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(255 * img), 0.6, heatmap_color, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================================
# LIME FOR TEXT EXPLANATION
# ============================================================================

def generate_lime_text_explanation(text, model, vocab, device, max_len=50):
    """Generate LIME-like explanation for text."""
    
    words = text.lower().split()
    if len(words) == 0:
        return None, []
    
    def preprocess_text(text, vocab, max_len=50):
        tokens = text.lower().split()
        ids = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    text_tensor = preprocess_text(text, vocab, max_len)
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
        
        perturbed_tensor = preprocess_text(perturbed_text, vocab, max_len)
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

def visualize_lime_explanation(word_importance):
    """Visualize LIME text explanation."""
    
    if not word_importance:
        return None
    
    words, importances = zip(*word_importance)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Color mapping
    colors = ['red' if imp > 0.7 else 'orange' if imp > 0.4 else 'lightgray' for imp in importances]
    
    # Horizontal bar chart
    y_pos = np.arange(len(words))
    ax.barh(y_pos, importances, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.set_xlabel('Importance')
    ax.set_title('LIME: Word Importance for Prediction')
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    return fig

# ============================================================================
# GENERATE EXPLANATIONS FOR SAMPLES
# ============================================================================

# Get Layer 0 (the only compatible layer)
def get_layer0(model):
    for name, child in model.vision.named_children():
        if name == '0':
            return child
    return None

layer0 = get_layer0(model)
print(f"\n✅ Layer 0 found: {layer0}")

# Initialize Layer-Wise Grad-CAM
layer_cam = LayerWiseGradCAM(model, layer0, DEVICE)
print("✅ Layer-wise Grad-CAM initialized!")

# Process each sample
for idx, sample in enumerate(samples[:2]):  # Process first 2 samples
    image = sample['image']
    label = sample['label']
    image_id = sample['image_id']
    text = sample['text']
    
    print(f"\n{'='*60}")
    print(f"📊 SAMPLE {idx+1}: {image_id} (True: {label})")
    print(f"   Text: {text[:100]}..." if len(text) > 100 else f"   Text: {text}")
    print(f"{'='*60}")
    
    try:
        # Generate Grad-CAM heatmap
        heatmap, target_class, loss = layer_cam.generate_layer_heatmaps(image, text)
        
        if heatmap is not None:
            class_label = 'Happy' if target_class == 0 else 'Sad'
            print(f"   Predicted: {target_class} ({class_label})")
            print(f"   Heatmap shape: {heatmap.shape}")
            print(f"   Max activation: {np.max(heatmap):.4f}")
            print(f"   Mean activation: {np.mean(heatmap):.4f}")
            
            # Visualize heatmap
            fig = visualize_heatmap(image, heatmap, f"Layer 0 Grad-CAM\nPredicted: {class_label}")
            plt.suptitle(f'Sample: {image_id} (True: {label})', fontsize=12)
            plt.tight_layout()
            plt.show()
            
            # Save
            save_path = os.path.join(MODEL_DIR, f'gradcam_{image_id}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            print("   ❌ Grad-CAM generation failed")
        
        # ============================================================
        # LIME Text Explanation
        # ============================================================
        print(f"\n📝 LIME Text Explanation:")
        
        if text.strip():
            base_pred, word_importance = generate_lime_text_explanation(
                text, model, vocab, DEVICE, max_len=50
            )
            
            if word_importance and len(word_importance) > 0:
                class_names = ['Happy', 'Sad']
                print(f"   Text-based prediction: {class_names[base_pred]}")
                print(f"   Word importance:")
                for word, imp in sorted(word_importance, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"      {word}: {imp:.4f}")
                
                # Visualize LIME
                fig2 = visualize_lime_explanation(word_importance)
                if fig2:
                    plt.show()
                    fig2.savefig(os.path.join(MODEL_DIR, f'lime_{image_id}.png'), dpi=300, bbox_inches='tight')
                    print(f"✅ LIME saved: {MODEL_DIR}lime_{image_id}.png")
            else:
                print("   ❌ LIME explanation not available")
        else:
            print("   ❌ No text provided for LIME")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("📋 LAYER-WISE GRAD-CAM WITH LIME - SUMMARY")
print("="*60)

print("\n✅ COMPONENTS IMPLEMENTED:")
print("   1. Layer-wise Grad-CAM using Layer 0 (features.0.0)")
print("   2. LIME text explanation for self-reflections")
print("   3. Comprehensive visualization for each sample")

print("\n📁 OUTPUTS SAVED:")
for f in os.listdir(MODEL_DIR):
    if f.startswith('gradcam_') or f.startswith('lime_'):
        print(f"      - {f}")

print("\n💡 UNDERSTANDING THE OUTPUT:")
print("   - Grad-CAM heatmap shows which image regions influenced prediction")
print("   - LIME explanation shows which words were most important")
print("   - Layer 0 captures basic features (edges, colors, simple patterns)")
print("   - Since MobileNetV2 uses depthwise separable convolutions, only Layer 0 is compatible with Grad-CAM")

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
