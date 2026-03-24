# 📸 Image_Caption_Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-BLIP-yellow)

An advanced, multi-modal AI system that generates accurate, contextual, and engaging captions for images. This project demonstrates a practical approach to solving the **"Domain Gap"** in Deep Learning by combining custom-trained academic models with State-of-the-Art (SOTA) Transformers and Generative AI.

---

## 🧠 The Engineering Problem & Solution
Standard image captioning models trained on limited datasets (like Flickr8k) struggle significantly with out-of-domain images such as digital art, anime, or text-heavy memes (The Domain Gap). 

To solve this hardware and data limitation, this project implements a **Hybrid Pipeline Architecture**:
1. It trains a base **CNN-LSTM model** from scratch to establish foundational deep learning concepts.
2. It integrates **Salesforce's BLIP** (trained on 14M+ images) to bridge the domain gap.
3. It utilizes **Google's Gemini LLM** for rich, human-like contextualization and formal formatting.

---

## 🏗️ System Architecture Pipeline

The inference pipeline runs sequentially through four major AI systems:

1. **Vision Detection Layer (YOLOv8):** Scans the image to detect and list distinct real-world entities along with confidence scores.
2. **Feature Extraction & Decoding (VGG16 + LSTM):** The custom base model. VGG16 extracts a 4096-dimensional feature vector, which is decoded into a sequence of words using an LSTM network with Beam Search optimization.
3. **Domain Adaptation (BLIP Transformer):** Acts as a highly accurate fallback/comparator for complex visual inputs that the base model cannot comprehend.
4. **Contextualization Engine (Gemini LLM):** Performs detailed visual analysis to generate formal bilingual descriptions (English & Hindi) with perfect grammar and context.

---

## ✨ Key Features
* **Multi-Model Dashboard:** A sleek, interactive Streamlit UI comparing outputs from different AI architectures simultaneously.
* **Bilingual Translation:** Real-time Neural Translation of generated captions into Hindi.
* **Beam Search Algorithm:** Implemented custom Beam Search (width=3) instead of Greedy Search for optimal sentence formulation.
* **RAM-Optimized Training:** Custom sequence-based Python Generators built to train the CNN-LSTM model locally without choking system memory.

---

## 🛠️ Tech Stack
* **Deep Learning Framework:** TensorFlow, Keras, PyTorch
* **Computer Vision:** OpenCV, Ultralytics (YOLOv8), PIL
* **Pre-trained Models:** VGG16, Salesforce BLIP (HuggingFace Transformers)
* **Generative AI:** Google Gemini API
* **Frontend/Deployment:** Streamlit
* **NLP & Utilities:** NLTK (BLEU Scores), Deep-Translator

---

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/Image_Caption_Generator.git](https://github.com/your-username/Image_Caption_Generator.git)
cd Image_Caption_Generator
