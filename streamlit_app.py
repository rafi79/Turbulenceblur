import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
import PIL.Image
import io

class TextExtractor:
    def __init__(self, api_key='AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E'):
        genai.configure(api_key=api_key)
        # Using pro-vision model for better text recognition
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def extract_text(self, image):
        """Extract text from turbulent image using Gemini"""
        try:
            # Convert to PIL Image
            if isinstance(image, np.ndarray):
                image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            prompt = """
            Please read and extract only the text from this image that has been affected by atmospheric turbulence.
            Format:

            Extracted Text:
            [Write the exact text you can read, maintaining line breaks as they appear]

            Confidence: [High/Medium/Low]

            If any parts are unclear, indicate with [...].
            Focus only on the text content, no other descriptions needed.
            """
            
            response = self.model.generate_content([prompt, image])
            response.resolve()
            
            return response.text
            
        except Exception as e:
            return f"Text extraction error: {str(e)}"

def main():
    st.title("Turbulent Text Extractor")
    
    # Sidebar for API key
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E",
        type="password"
    )
    
    # Initialize extractor
    extractor = TextExtractor(api_key)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload a turbulent image with text", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader("Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                # Extract text
                extracted_text = extractor.extract_text(image)
                
                # Display results
                st.subheader("Extracted Text")
                st.text_area("Text Content", extracted_text, height=200)
                
                # Copy button
                st.download_button(
                    "Download Text",
                    data=extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
