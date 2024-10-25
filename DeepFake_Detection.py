import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import random
import os

# Set random seed for reproducibility
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

# Load pre-trained model (ResNet50, modified)
NPRmodel = resnet50()
NPRmodel.fc1 = nn.Linear(512, 1)
NPRmodel.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
del NPRmodel.layer3, NPRmodel.layer4, NPRmodel.fc

# Load model weights
@st.cache_data
#@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_model():
    NPR_url = "C:\\Users\\darth\\Desktop\\CV code Demo\\NPR-DeepfakeDetection\\model_epoch_last_3090.pth"#'https://raw.githubusercontent.com/chuangchuangtan/NPR-DeepfakeDetection/main/model_epoch_last_3090.pth'
    state_dict = torch.hub.load_state_dict_from_url(NPR_url, map_location='cpu')
    NPRmodel.load_state_dict(state_dict, strict=True)
    NPRmodel.eval()
    return NPRmodel

# Image transformation
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Interpolation function
def interpolate(img, factor):
    return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True),
                         scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

# Predict whether image is fake or real
def predict_image(img, model):
    img = trans(img.convert('RGB')).unsqueeze(0)
    _, c, w, h = img.shape
    if w % 2 == 1: img = img[:, :, :-1, :]
    if h % 2 == 1: img = img[:, :, :, :-1]
    
    NPR = img - interpolate(img, 0.5)
    with torch.no_grad():
        x = model.conv1(NPR * 2.0 / 3.0)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x).mean(dim=(2, 3), keepdim=False)
        x = model.fc1(x)
        pred = x.sigmoid().cpu().numpy()
    
    return {'Fake Image': float(pred), 'Real Image': 1.0 - float(pred)}

# Streamlit app interface
st.title("NPR Deepfake Detection")

# Input selection (URL or upload)
option = st.radio("Select Input Type", ('Upload Image', 'Image URL'))
if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        model = load_model()
        result = predict_image(image, model)
        st.write("Prediction: ", result)
        
elif option == 'Image URL':
    url = st.text_input("Enter Image URL")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='Image from URL', use_column_width=True)
            model = load_model()
            result = predict_image(image, model)
            st.write("Prediction: ", result)
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")
