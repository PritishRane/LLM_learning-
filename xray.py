import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from PIL import Image
import io
import torch
from torchvision import transforms
from torchvision.models import resnet50

# Load environment variables
_ = load_dotenv(find_dotenv())
groq_api_key = os.environ["XXXXX"]

# Initialize the Groq LLM with a specific model
groq_model = "llama3-70b-8192"  # High-performing Groq model
llm = ChatGroq(model=groq_model)

# Load a pre-trained image classification model for better analysis
model = resnet50(pretrained=True)
model.eval()

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Define a prompt template
prompt = PromptTemplate(
    input_variables=["description"],
    template="You are an AI medical assistant specialized in X-ray interpretation. Analyze the following X-ray image description and provide insights: {description}"
)

# Create an LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("X-ray Image Detection AI (LangChain + Groq + Streamlit)")
st.write("Upload an X-ray image and get AI-powered analysis!")

# Image upload
uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Processing image using ML model
    st.write("Processing image...")
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    predicted_label = f"Predicted class index: {predicted.item()}"  # Needs mapping to actual class names
    
    # Generating a detailed medical description
    image_description = f"The uploaded X-ray shows {predicted_label}. Please provide medical insights based on this observation."
    response = llm_chain.run(image_description)
    
    st.write("### X-ray Analysis:")
    st.write(response)
