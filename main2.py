import streamlit as st
import numpy as np
from PIL import Image
from fpdf import FPDF
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# Define path to your local model
weights_path = "colorectal_cancer_model.h5"

# Define class labels
class_labels = ['Adenocarcinoma', 'High-grade IN', 'Low-grade IN', 'Normal', 'Polyp', 'Serrated adenoma']

# Build the model
def build_model():
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_labels), activation='softmax')
    ])
    return model

# Load model
try:
    model = build_model()
    model.load_weights(weights_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Preprocess the uploaded image
def preprocess_image(image, img_size=(224, 224)):
    image = image.resize(img_size)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Predict using local model
def predict_category(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image, verbose=0)[0]
    predicted_class_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_class_index]
    return predicted_label

# Information about the stages and precautions
stage_info = {
    'Low-grade IN': {
        'description': "Low-grade intraepithelial neoplasia (IN) refers to mildly abnormal cells that are not cancerous.",
        'precautions': "Regular check-ups and early monitoring can help prevent progression."
    },
    'Adenocarcinoma': {
        'description': "Adenocarcinoma is a type of cancer that begins in the glandular cells of the colon or rectum.",
        'precautions': "Treatment options include surgery, chemotherapy, and radiation. Consult a medical professional for further guidance."
    },
    'High-grade IN': {
        'description': "High-grade intraepithelial neoplasia refers to more abnormal cells, which have a higher risk of becoming cancerous.",
        'precautions': "Close monitoring and potential intervention may be required to prevent cancer development."
    },
    'Normal': {
        'description': "No signs of cancer or precancerous conditions are observed. Regular screening is recommended.",
        'precautions': "Maintain a healthy lifestyle, eat a balanced diet, and undergo regular screenings."
    },
    'Polyp': {
        'description': "Polyps are growths on the inner lining of the colon or rectum that can become cancerous over time.",
        'precautions': "Polyp removal is recommended to prevent cancer. Regular colonoscopies are advised."
    },
    'Serrated adenoma': {
        'description': "Serrated adenomas are abnormal growths in the colon that have the potential to turn cancerous.",
        'precautions': "Removal of serrated adenomas is necessary to prevent cancer. Follow-up care is important."
    }
}

def generate_pdf_report(mapped_class, stage_info):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(200, 10, txt="Cancer Detection Report", ln=True, align="C")
    pdf.ln(10)

    # Detected class
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Detected Class: {mapped_class}", ln=True)

    # Add stage information and precautions
    if mapped_class in stage_info:
        pdf.ln(5)
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(200, 10, txt="Stage Information:", ln=True)
        
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"Description: {stage_info[mapped_class]['description']}")
        pdf.ln(2)
        pdf.multi_cell(0, 10, txt=f"Precautions: {stage_info[mapped_class]['precautions']}")
    else:
        pdf.ln(5)
        pdf.cell(200, 10, txt="No additional information available.", ln=True)

    # Output the PDF to bytes
    pdf_output = pdf.output(dest="S").encode("latin1")
    return pdf_output

# Streamlit app interface
st.title("Colorectal Cancer Classification", anchor="colorectal-cancer")

# File uploader to allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Perform inference
    st.write("Detecting...")
    predicted_label = predict_category(image)

    if predicted_label:
        st.info(f"Detected Class: {predicted_label}")

        # Display stage information and precautions
        if predicted_label in stage_info:
            st.write(f"### Stage Information:")
            st.write(f"**Description**: {stage_info[predicted_label]['description']}")
            st.write(f"**Precautions**: {stage_info[predicted_label]['precautions']}")
        else:
            st.write("No additional information available for this class.")

        # Enable report download
        if st.button("Download Report"):
            pdf = generate_pdf_report(predicted_label, stage_info)
            st.download_button("Download PDF Report", data=pdf, file_name="cancer_report.pdf", mime="application/pdf")
    else:
        st.write("No predictions were found.")
else:
    st.write("Upload an image to get started.")
