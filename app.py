import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import googlenet
from PIL import Image

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="Phising Website Screenshot Classification",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Phising Website Screenshot Classification")
st.caption("Model: GoogLeNet (trained on screenshots)")

# --------------------------------
# Load model
# --------------------------------
@st.cache_resource
def load_model():
    model = googlenet(pretrained=False, aux_logits=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    state_dict = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# --------------------------------
# Image preprocessing
# --------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------------------------------
# Upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload a screenshot",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width='stretch')

    if st.button("🔍 Analyze"):
        with st.spinner("Running model..."):
            x = transform(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0]

        legitimate_prob = probs[0].item() * 100
        phishing_prob   = probs[1].item() * 100

        st.subheader("📊 Prediction")

        col1, col2 = st.columns(2)
        col1.metric("✅ Legitimate", f"{legitimate_prob:.2f}%")
        col2.metric("🚨 Phishing", f"{phishing_prob:.2f}%")

        if phishing_prob / 100 > 0.3:
            st.error("🚨 This screenshot is likely PHISHING")
        else:
            st.success("✅ This screenshot looks LEGITIMATE")
