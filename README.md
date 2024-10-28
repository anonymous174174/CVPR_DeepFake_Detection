# CVPR DeepFake Detection

## Introduction

This repository contains a demo of a DeepFake detection model based on the research paper and is not its official implementation [link](https://arxiv.org/abs/2312.10461#).

## Requirements

- Python 3.8 or later
- PyTorch 2.4.1
- Torchvision 0.19.1
- Streamlit 1.38.0
- OpenCV 4.10.0.84
- Pillow 10.4.0
- Numpy 2.1.1


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anonymous174174/CVPR_DeepFake_Detection.git
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Navigate to the repository directory:
   ```bash
   cd CVPR_DeepFake_Detection
   ```

4. Run the model:
   ```bash
   streamlit run DeepFake_Detection.py
   ```

5. Open a web browser and go to [http://localhost:8501](http://localhost:8501) to access the Streamlit app.

6. Upload an image or enter an image URL to test the model. The model will predict whether the image is real or fake and display the result.
