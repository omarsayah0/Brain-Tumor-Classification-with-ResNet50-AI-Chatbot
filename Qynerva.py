import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import gdown
import os
from brain_tumors import set_data
from mistralai import Mistral
from check_mri import is_mri_clip

#python -m streamlit run Qynerva.py
mistral_client = Mistral(api_key="Lxbe1yGj29G2kXj2WH996CAgl9lBApia")
mistral_model = "mistral-large-latest"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "mistral_response" not in st.session_state:
    st.session_state.mistral_response = None

def get_mistral_answer(result=None, Confidence = None, state=False, user_input=""):
    try:
        messages = st.session_state.chat_history.copy()
        if state and result is not None:
            user_message = {
                "role": "user",
                "content": (
                    "first of all you are Qynerva**, a friendly AI assistant for brain MRI scans. "
                    "you help user understand what's going on in your scan using smart AI, but remember – "
                    "**you are not a real doctor**, so always the user have tocheck with a healthcare professional for medical decisions"
                    "dont mention that you are Qynerva to the user he already knows that , but just in case if he asked you"
                    "The user uploaded an MRI scan. "
                    f"The model predicted: {result} and the Confidence is {Confidence:.2f}%. "
                    "Please tell him what iss the results and explain what that means, and share your findings first."
                    "use emojies in your response."
                )
            }
            if st.session_state.mistral_response is not None:
                return st.session_state.mistral_response
        else:
            user_message = {
                "role": "user",
                "content": user_input
            }

        messages.append(user_message)

        with st.spinner("Qynerva is thinking..."):
            response = mistral_client.chat.complete(
                model=mistral_model,
                messages=messages
            )

        bot_response = response.choices[0].message.content.strip()

        if state:
            st.session_state.mistral_response = bot_response

        st.session_state.chat_history.append(user_message)
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        return bot_response

    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_resource
def load_model():
    model_filename = 'brain_tumor_resnet50_model.keras'
    if not os.path.exists(model_filename):
        url = 'https://www.dropbox.com/scl/fi/d42ee2snyn69xuw7lv2ch/brain_tumor_resnet50_model.keras?rlkey=vul1cr9qi4ierv387y6puf43j&st=7frpih65&dl=1'
        gdown.download(url, model_filename, quiet=False)
    model = keras.models.load_model(model_filename, compile=False)
    return model

def main():
    st.title("Qynerva🤖")

    with st.sidebar:
        st.header("Chat History")
        if st.session_state.chat_history:
            for i in range(0, len(st.session_state.chat_history), 2):
                user_msg = st.session_state.chat_history[i]
                assistant_msg = st.session_state.chat_history[i+1] if i+1 < len(st.session_state.chat_history) else None
                with st.expander(user_msg["content"], expanded=False):
                    if assistant_msg:
                        st.write(assistant_msg["content"])
        else:
            st.write("No messages yet.")

    model = load_model()
    train_generator, val_generator = set_data()
    st.header("Hi there! 👋 I'm **Qynerva** 🤖, your friendly AI assistant for brain MRI scans. I help you understand what's going on in your scan using smart AI, but remember"
            "**I'm not a real doctor**, so always check with a healthcare professional for medical decisions 🧠💬")
    
    st.warning("""
    **Warning ⚠️:**  
    This AI model is still under development. If the uploaded MRI contains another brain disease or abnormality not listed below, the model may produce inaccurate or random predictions.  
    Currently, the model is only trained to detect the following cases:
    - **Normal brain MRI**
    - **Pituitary tumor**
    - **Meningioma tumor**
    - **Glioma tumor**
               
    We are actively working on improving the model by including more disease types. Stay tuned for updates in the near future!
    """)

    st.header("Upload Your MRI Image")
    uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"])

    if "interpretation_shown" not in st.session_state:
        st.session_state.interpretation_shown = False

    if uploaded_file is not None and st.session_state.prediction_result is None:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((256, 256))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)


        check = is_mri_clip(image)

        if check == True :
            preds = model.predict(img_array, verbose=0)
            class_labels = list(val_generator.class_indices.keys())
            pred_label_idx = np.argmax(preds)
            pred_class = class_labels[pred_label_idx]
            confidence = float(np.max(preds)) * 100
        else :
            st.error("🚫 The uploaded image does not appear to be an MRI scan for the Brain. Please upload a valid brain MRI image for the Brain.")
            return
    
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded MRI", use_container_width=True)

        with col2:
            st.image(image, caption="Another Image", use_container_width=True)


        interpretation = get_mistral_answer(result=pred_class, Confidence = confidence, state=True)
        st.write("**Qynerva🤖:**")
        st.write(interpretation)

    st.divider()
        
    st.header("Chat section")

    st.markdown("### Your Message:")
    user_message = st.text_input("", key="user_input")
    if st.button("Send"):
        if user_message.strip():
            assistant_response  = get_mistral_answer(user_input=user_message, state=False)
            st.markdown(f"**👱🏻:** {user_message}")
            st.markdown(f"**Qynerva🤖:** {assistant_response}")

if __name__ == "__main__":
    main()