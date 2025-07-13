# StudyBuddy AI 🤖

An AI-powered learning assistant optimized with Intel's OpenVINO™ toolkit. This project allows users to ask questions or summarize text using either typed input or speech-to-text, all running on an efficient, quantized language model.

---

## 🌟 Features

- **Interactive Q&A:** Ask complex questions and get concise answers from the AI.
- **Text Summarization:** Paste long articles or documents and receive a bullet-point summary.
- **Speech-to-Text Input:** Use your voice to ask questions via a simple "Listen" button.
- **Highly Optimized:** The backend runs a `TinyLlama` model, aggressively optimized with OpenVINO INT4 quantization for maximum efficiency and a small footprint on CPU. This makes the model faster and significantly smaller.

## 🛠️ Technology Stack & Key Concepts

- **AI Model:** `TinyLlama-1.1B-Chat-v1.0`
- **Optimization Toolkit:** Intel® OpenVINO™
- **Optimization Techniques:**
    - **FP16 Conversion:** The initial conversion from PyTorch to a performance-oriented OpenVINO format using the `optimum-cli` tool.
    - **INT4 Quantization:** Aggressive weight compression using the `NNCF` (Neural Network Compression Framework) to significantly reduce model size and accelerate inference speed.
- **UI Framework:** Gradio
- **Speech Recognition:** `SpeechRecognition` library (leveraging Google's Speech-to-Text API).
- **Deployment:** The live demo is hosted on Hugging Face Spaces. This GitHub repository contains the source code and instructions to replicate the project.

## 🚀 How to Run This Project Locally

This repository contains the source code for the application. The large AI model files are **not** included here. The following steps will guide you through generating the necessary model files and running the application on your own machine.

### 1. Prerequisites
- Python 3.9+
- Git installed on your system.

### 2. Clone the Repository
Open your terminal and clone this project.
```bash
git clone https://github.com/sparuchu/StudyBuddy-AI.git
cd StudyBuddy-AI
```

### 3. Setup Environment and Install Dependencies
It is highly recommended to use a Python virtual environment.
```bash
# Create a virtual environment
python -m venv env

# Activate it (on Windows Command Prompt)
.\env\Scripts\activate.bat

# Install all required packages from the requirements file
pip install -r requirements.txt
```

### 4. Prepare the AI Model
The application requires an optimized OpenVINO model. The following two commands will generate it.

**First, download the base model and convert it to OpenVINO FP16 format.** This command uses `optimum-cli` to perform the conversion. Note that this step will download a large file (~2GB).
```bash
optimum-cli export openvino --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" tinyllama_fp16
```

**Next, run the quantization script to apply INT4 optimization.** This script reads the FP16 model and creates the final, highly-compressed `tinyllama_int8` folder that the application uses.
```bash
python quantize.py
```

### 5. Launch the Application
Once the `tinyllama_int8` folder has been created by the previous steps, you are ready to launch the Gradio web interface.
```bash
python app.py
```
Open the local URL provided in the terminal (e.g., `http://127.0.0.1:7860`) in your web browser to use the application.
