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

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🎨",
    layout="wide"
)

st.title("Emotion Detection from Children's Drawings")
st.markdown("---")


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


def create_vocab(vocab_size=3423):
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


def find_model_file():
    search_paths = [
        "Emotion_Models/MM_MobileNetV2_BiLSTM_final.pt",
        "MM_MobileNetV2_BiLSTM_final.pt",
        "*.pt",
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
        if '*' in path:
            matches = glob.glob(path)
            if matches:
                return matches[0]
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pt'):
                return os.path.join(root, file)
    return None


@st.cache_resource
def load_cached_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = find_model_file()
    if model_path is None:
        st.error("Model file not found")
        return None, None, device
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        embedding_weight = state_dict.get('text_encoder.embedding.weight')
        vocab_size = embedding_weight.shape[0] if embedding_weight is not None else 3423
        vocab = create_vocab(vocab_size)
        text_encoder = BiLSTMTextEncoder(vocab_size)
        model = MultimodalModel('mobilenet_v2', text_encoder)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        model.eval()
        return model, vocab, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, device


def get_feature_layer_5(model):
    vision = model.vision
    if hasattr(vision, 'features') and hasattr(vision.features, '_modules'):
        keys = list(vision.features._modules.keys())
        if keys:
            for key in reversed(keys):
                module = vision.features._modules[key]
                if isinstance(module, nn.Conv2d):
                    return module
            return vision.features._modules[keys[-1]]
    for name, module in vision.named_modules():
        if isinstance(module, nn.Conv2d):
            return module
    return vision


def generate_gradcam_heatmap(model, image, target_class):
    device = next(model.parameters()).device
    target_layer = get_feature_layer_5(model)
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


def overlay_heatmap_jet(image, heatmap, alpha=0.5):
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
    img = tensor.cpu().detach().numpy()
    if img.ndim == 4:
        img = img.squeeze(0)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img


def main():
    model, vocab, device = load_cached_model()
    if model is None:
        st.stop()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Upload")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        child_text = st.text_area(
            "Child's Explanation (Optional)",
            placeholder="What did the child say about their drawing?",
            height=60,
            label_visibility="collapsed"
        )

        predict_button = st.button("Predict", type="primary", use_container_width=True)

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

                        emotion = "Happy" if pred == 0 else "Sad"
                        confidence = probs[0][pred].item()

                        st.session_state['emotion'] = emotion
                        st.session_state['confidence'] = confidence
                        st.session_state['current_image'] = img_tensor
                        st.session_state['current_pred'] = pred
                        st.session_state['current_pil'] = image

                        st.success(f"{emotion} — {confidence:.1%} confidence")

                    except Exception as e:
                        st.error(f"Error: {e}")

    with col_right:
        st.subheader("Explanation")

        if 'current_image' in st.session_state and st.session_state.get('current_image') is not None:
            image_tensor = st.session_state['current_image']
            pred_class = st.session_state['current_pred']

            alpha = st.slider("Heatmap Opacity:", 0.1, 1.0, 0.5, 0.1)

            with st.spinner("Generating explanation..."):
                try:
                    heatmap = generate_gradcam_heatmap(model, image_tensor, pred_class)
                    img_display = normalize_image(image_tensor)
                    overlay = overlay_heatmap_jet(img_display, heatmap, alpha)

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.image(img_display, caption="Original", use_container_width=True)

                    with col_b:
                        st.image(heatmap, caption="Heatmap", use_container_width=True, clamp=True)

                    with col_c:
                        st.image(overlay, caption="Overlay", use_container_width=True)

                except Exception as e:
                    st.error(f"Error generating explanation: {e}")

                # Prediction output below explanation - BIG and visible
                emotion = st.session_state.get('emotion', '')
                confidence = st.session_state.get('confidence', 0)

                st.markdown("---")

                if emotion == "Happy":
                    st.markdown(
                        f"""
                        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;">
                            <h1 style="font-size: 48px; margin: 0; color: #155724;">HAPPY</h1>
                            <p style="font-size: 24px; margin: 5px 0 0 0; color: #155724;">{confidence:.1%} confidence</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="background-color: #f8d7da; padding: 20px; border-radius: 10px; text-align: center;">
                            <h1 style="font-size: 48px; margin: 0; color: #721c24;">SAD</h1>
                            <p style="font-size: 24px; margin: 5px 0 0 0; color: #721c24;">{confidence:.1%} confidence</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("Upload an image and click 'Predict' to see the explanation.")


if __name__ == "__main__":
    main()
