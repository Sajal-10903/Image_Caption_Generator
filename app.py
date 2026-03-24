import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from ultralytics import YOLO
from deep_translator import GoogleTranslator
import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Hybrid Image Captioning System", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Hybrid Image Captioning System")
st.write("**Architecture Pipeline:** YOLOv8 (Detection) ➡️ VGG16 (Features) ➡️ LSTM Beam Search (Caption) ➡️ BLIP (Domain Adaptation) ➡️ Gemini LLM")

@st.cache_resource
def load_all_models():
    inputs1 = tf.keras.layers.Input(shape=(4096,))
    fe1 = tf.keras.layers.Dropout(0.4)(inputs1)
    fe2 = tf.keras.layers.Dense(256, activation='relu')(fe1)
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 35

    inputs2 = tf.keras.layers.Input(shape=(max_length,))
    se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = tf.keras.layers.Dropout(0.4)(se1)
    se3 = tf.keras.layers.LSTM(256)(se2)

    decoder1 = tf.keras.layers.add([fe2, se3])
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    caption_model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    caption_model.load_weights('image_captioning_model.h5')
    
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    
    yolo_model = YOLO('yolov8n.pt')
    
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    return caption_model, tokenizer, vgg_model, yolo_model, blip_processor, blip_model

model, tokenizer, vgg_model, yolo_model, blip_processor, blip_model = load_all_models()
max_length = 35 

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def get_multiple_captions(model, image_features, tokenizer, max_length, beam_width=3):
    start_seq = tokenizer.texts_to_sequences(['startseq'])[0]
    sequences = [(start_seq, 0.0)] 
    
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == tokenizer.word_index.get('endseq'):
                all_candidates.append((seq, score))
                continue
            
            padded_seq = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([image_features, padded_seq], verbose=0)[0]
            
            top_k_indices = np.argsort(preds)[-beam_width:]
            for word_idx in top_k_indices:
                new_seq = seq + [word_idx]
                new_score = score + np.log(preds[word_idx] + 1e-10)
                all_candidates.append((new_seq, new_score))
        
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
        
        if all(s[0][-1] == tokenizer.word_index.get('endseq') for s in sequences):
            break
            
    results = []
    for seq, score in sequences:
        words = [idx_to_word(idx, tokenizer) for idx in seq if idx not in [tokenizer.word_index.get('startseq'), tokenizer.word_index.get('endseq')]]
        caption = ' '.join([w for w in words if w is not None]).capitalize()
        
        seq_len = len(words) if len(words) > 0 else 1
        confidence = np.exp(score / seq_len) * 100
        
        if caption.strip() and caption not in [r[0] for r in results]:
            results.append((caption, round(confidence, 2)))
            
    return results

tab1, tab2 = st.tabs(["Live Analysis Dashboard", "Session History"])

with tab1:
    st.markdown("### Input Layer")
    uploaded_file = st.file_uploader("Upload Image for Multi-Stage Analysis", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        if st.button("Run Hybrid Inference Pipeline", use_container_width=True):
            with st.spinner("Processing: YOLO Detection ➡️ VGG16 Extraction ➡️ LSTM Decoding ➡️ LLM Contextualization..."):
                
                img_cv = np.array(image)
                results = yolo_model.predict(img_cv)
                yolo_annotated_img = results[0].plot() 
                
                detected_objects = []
                for box in results[0].boxes:
                    cls_name = yolo_model.names[int(box.cls[0])]
                    conf = float(box.conf[0]) * 100
                    detected_objects.append(f"{cls_name.capitalize()} ({conf:.1f}%)")
                detected_objects = list(set(detected_objects))

                image_resized = image.resize((224, 224))
                image_arr = img_to_array(image_resized)
                image_arr = image_arr.reshape((1, image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
                image_arr = preprocess_input(image_arr)
                feature = vgg_model.predict(image_arr, verbose=0)
                
                caption_options = get_multiple_captions(model, feature, tokenizer, max_length, beam_width=3)
                best_caption, best_confidence = caption_options[0]
                
                hindi_caption_base = GoogleTranslator(source='en', target='hi').translate(best_caption)

                blip_inputs = blip_processor(image, return_tensors="pt")
                blip_out = blip_model.generate(**blip_inputs)
                blip_caption = blip_processor.decode(blip_out[0], skip_special_tokens=True).capitalize()
                
                hindi_caption_blip = GoogleTranslator(source='en', target='hi').translate(blip_caption)
                
                st.session_state.history.append({
                    "image": image,
                    "yolo_image": yolo_annotated_img,
                    "caption": best_caption,
                    "blip_caption": blip_caption,
                    "confidence": best_confidence,
                    "hindi": hindi_caption_base
                })
                
                st.success("Analysis Complete.")
                
                st.subheader("Inference Visualization")
                img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
                with img_col2:
                    st.image(yolo_annotated_img, use_container_width=True, channels="BGR")
                
                st.markdown("---")
                
                col3, col4 = st.columns([1.5, 1])
                
                with col3:
                    st.markdown("#### Generative AI Contextualization")
                    try:
                        MY_GEMINI_KEY = "YOUR_API_KEY_HERE" 
                        genai.configure(api_key=MY_GEMINI_KEY)
                        llm = genai.GenerativeModel('gemini-2.5-flash')
                        
                        prompt = "Perform a detailed visual analysis. Identify primary subjects, background context, and any visible text. Provide one professional English description and one formal Hindi description. Output must be pure text without any emojis or decorative characters."
                        
                        response = llm.generate_content([prompt, image])
                        st.info(response.text)
                    except Exception as e:
                        st.error(f"LLM Integration Error: {e}")

                    st.markdown("---")

                    st.subheader("Secondary Prediction (BLIP SOTA Transformer)")
                    st.success(f"**English:** {blip_caption}")
                    st.success(f"**Hindi:** {hindi_caption_blip}")

                    with st.expander("View Primary CNN-LSTM Prediction"):
                        st.info(f"**English:** {best_caption}")
                        st.info(f"**Hindi:** {hindi_caption_base}")
                        
                        st.markdown("#### Alternative Search Candidates")
                        for i, (cap, conf) in enumerate(caption_options[1:], start=1):
                            st.write(f"**Candidate {i+1}:** {cap} *(Score: {conf}%)*")
                
                with col4:
                    st.subheader("Performance Metrics")
                    st.metric(label="Model Confidence Score", value=f"{float(best_confidence):.2f}%")
                    safe_progress = min(max(float(best_confidence) / 100.0, 0.0), 1.0)
                    st.progress(safe_progress)
                    
                    st.markdown("**YOLO Objects Detected:**")
                    if len(detected_objects) > 0:
                        for obj in detected_objects:
                            st.markdown(f"- {obj}")
                    else:
                        st.write("No distinct objects identified in the current domain.")

with tab2:
    st.markdown("### Inference History")
    if len(st.session_state.history) == 0:
        st.write("Session history is currently empty.")
    else:
        for idx, item in enumerate(reversed(st.session_state.history)):
            st.markdown(f"**Log Entry #{len(st.session_state.history) - idx}**")
            h_col1, h_col2 = st.columns([1, 2])
            with h_col1:
                st.image(item["yolo_image"], use_container_width=True, channels="BGR")
            with h_col2:
                st.markdown(f"**CNN-LSTM Caption:** {item['caption']}")
                st.markdown(f"**BLIP Caption:** {item['blip_caption']}")
                st.markdown(f"**Confidence:** {item['confidence']}%")
            st.divider()