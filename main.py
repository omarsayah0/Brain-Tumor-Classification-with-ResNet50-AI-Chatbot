import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import json
from tensorflow import keras
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from brain_tumors import set_data
from mistralai import Mistral
import requests

def download_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

api_key = st.secrets["api_keys"]["mistral"]

mistral_model = "mistral-large-latest"

mistral_client = Mistral(api_key=api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "mistral_response" not in st.session_state:
    st.session_state.mistral_response = None

def get_mistral_answer(result=None, state=False, user_input=""):
    try:
        messages = st.session_state.chat_history.copy()
        if state and result is not None:
            user_message = {
                "role": "user",
                "content": (
                    "The user uploaded an MRI scan. "
                    f"The model predicted: {result}. "
                    "Please explain what that means, and share your findings first."
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

        with st.spinner("Chatbot is thinking..."):
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
    model_filename = 'brain_model.keras'

    if not os.path.exists(model_filename):
        file_id = '1lqn-dk20___Xq_JTH88DEP4VtZ9R4yka'
        download_from_google_drive(file_id, model_filename)

    model = keras.models.load_model(model_filename, compile=False)
    return model

@st.cache_data
def load_history():
    with open("training_history.json", "r") as f:
        history_data = json.load(f)
    return history_data

def plot_metric(history_data, metric, fine_tune_metric, title, ylabel):
    fig = plt.figure(figsize=(8, 4))
    epochs = range(len(history_data[metric]))
    plt.plot(epochs, history_data[metric], label=f'Training {ylabel} (Phase 1)')
    plt.plot(epochs, history_data[f'val_{metric}'], label=f'Validation {ylabel} (Phase 1)')

    fine_epochs = range(len(history_data[fine_tune_metric]))
    plt.plot(fine_epochs, history_data[fine_tune_metric], label=f'Training {ylabel} (Fine-tuning)')
    plt.plot(fine_epochs, history_data[f'fine_tune_val_{metric}'], label=f'Validation {ylabel} (Fine-tuning)')

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

def plot_class_distribution(train_generator, val_generator):
    train_labels = train_generator.classes
    val_labels = val_generator.classes
    class_labels = list(train_generator.class_indices.keys())

    train_counts = np.bincount(train_labels)
    val_counts = np.bincount(val_labels)
    x = np.arange(len(class_labels))

    fig = plt.figure(figsize=(6, 4))
    plt.bar(x - 0.35/2, train_counts, 0.35, label='Training')
    plt.bar(x + 0.35/2, val_counts, 0.35, label='Validation')
    plt.xticks(x, class_labels, rotation=45)
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Training and Validation Sets')
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

def plot_class_distribution_pie(train_generator):
    train_labels = train_generator.classes
    class_labels = list(train_generator.class_indices.keys())
    train_counts = np.bincount(train_labels)

    fig = plt.figure(figsize=(4, 4))
    plt.pie(train_counts, labels=class_labels, autopct='%1.1f%%', startangle=140)
    plt.title('Training Set Class Distribution')
    plt.axis('equal')
    st.pyplot(fig)

def show_predictions(model, val_generator, num_images=30):
    class_labels = list(val_generator.class_indices.keys())  
    images, labels = next(val_generator)
    num_images = min(num_images, len(images))

    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(12, 10))
    fig.suptitle("Sample Predictions from Validation Set")

    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            ax.axis("off")
            continue

        image = images[i]
        true_label = np.argmax(labels[i])
        pred_probs = model.predict(np.expand_dims(image, axis=0), verbose=0)
        pred_label = np.argmax(pred_probs)
        confidence = np.max(pred_probs) * 100

        ax.imshow(image.astype("uint8"))
        ax.set_title(
            f"True: {class_labels[true_label]}\nPred: {class_labels[pred_label]}\n({confidence:.2f}%)",
            color="green" if true_label == pred_label else "red", fontsize=8
        )
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)

def plot_confusion(val_generator, model):
    class_labels = list(val_generator.class_indices.keys())
    all_preds = []
    all_true = []

    for _ in range(len(val_generator)):
        images, labels = next(val_generator)
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(np.argmax(labels, axis=1))

    cm = confusion_matrix(all_true, all_preds)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax)
    st.pyplot(fig)

def main():
    st.title("Brain Tumor Classification & Chatbot")

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
    history_data = load_history()

    tab1, tab2 = st.tabs(["Classification & Chat", "Model Evaluation"])

    with tab1:
        st.header("Upload an MRI Image for Classification")
        uploaded_file = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"])

        if "interpretation_shown" not in st.session_state:
            st.session_state.interpretation_shown = False

        if uploaded_file is not None and st.session_state.prediction_result is None:
            image = Image.open(uploaded_file).convert("RGB")
            image = image.resize((256, 256))
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array, verbose=0)
            class_labels = list(val_generator.class_indices.keys())
            pred_label_idx = np.argmax(preds)
            pred_class = class_labels[pred_label_idx]
            confidence = float(np.max(preds)) * 100

            st.image(image, caption="Uploaded MRI", use_container_width =True)
            st.write(f"**Predicted Class:** {pred_class} (Confidence = {confidence:.2f}%)")

            interpretation = get_mistral_answer(result=pred_class, state=True)
            st.write("**🤖:**")
            st.write(interpretation)

        st.divider()
        
        st.header("Chat section")

        st.markdown("### Your Message:")
        user_message = st.text_input("", key="user_input")
        if st.button("Send"):
            if user_message.strip():
                assistant_response  = get_mistral_answer(user_input=user_message, state=False)
                st.markdown(f"**👱🏻:** {user_message}")
                st.markdown(f"**🤖:** {assistant_response}")

    with tab2:
        st.header("Model Evaluation")

        st.subheader("Accuracy & Loss across epochs")
        plot_metric(history_data, "accuracy", "fine_tune_accuracy", "Model Accuracy over Epochs", "Accuracy")
        plot_metric(history_data, "loss", "fine_tune_loss", "Model Loss over Epochs", "Loss")

        st.subheader("Class Distribution")
        plot_class_distribution(train_generator, val_generator)
        plot_class_distribution_pie(train_generator)

        st.subheader("Sample Predictions on the Validation Set")
        show_predictions(model, val_generator, num_images=30)

        st.subheader("Confusion Matrix")
        train_generator, val_generator = set_data()
        plot_confusion(val_generator, model)

if __name__ == "__main__":
    main()