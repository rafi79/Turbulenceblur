import streamlit as st
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import google.generativeai as genai
import PIL.Image
import io

class TextFocusedExtractor:
    def __init__(self, api_key='AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')
    
    def enhance_text(self, image):
        """Enhance text visibility in turbulent image"""
        # Convert to float32
        img_float = image.astype(np.float32) / 255.0
        
        # Text-focused preprocessing
        # 1. Strong bilateral filter to preserve edges
        denoised = cv2.bilateralFilter(
            (img_float * 255).astype(np.uint8),
            d=7,  # Smaller window for text
            sigmaColor=50,
            sigmaSpace=50
        ).astype(np.float32) / 255.0
        
        # 2. Local contrast enhancement
        if len(image.shape) == 3:
            lab = cv2.cvtColor((denoised * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            enhanced = enhanced.astype(np.float32) / 255.0
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply((denoised * 255).astype(np.uint8))
            enhanced = enhanced.astype(np.float32) / 255.0
        
        # 3. Text sharpening
        kernel_sharp = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]]) * 0.7  # Moderate sharpening
        
        if len(image.shape) == 3:
            sharpened = np.zeros_like(enhanced)
            for i in range(3):
                sharpened[..., i] = cv2.filter2D(enhanced[..., i], -1, kernel_sharp)
        else:
            sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # 4. Adaptive thresholding for text emphasis
        if len(image.shape) == 3:
            gray = cv2.cvtColor((sharpened * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (sharpened * 255).astype(np.uint8)
            
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Small block size for text
            2
        )
        
        # 5. Combine with original
        if len(image.shape) == 3:
            result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(
                (sharpened * 255).astype(np.uint8),
                0.7,  # Original weight
                result,
                0.3,  # Text emphasis weight
                0
            )
        else:
            result = cv2.addWeighted(
                (sharpened * 255).astype(np.uint8),
                0.7,
                thresh,
                0.3,
                0
            )
        
        return result
    
    def extract_text(self, image):
        """Extract text from enhanced image"""
        try:
            # Enhance image first
            enhanced = self.enhance_text(image)
            
            # Convert to PIL Image
            enhanced_pil = PIL.Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
            prompt = """
            Carefully read and extract the text from this image.
            The text may be a quote or saying.
            Return ONLY the exact text with proper line breaks and punctuation.
            Include any attribution or source if visible.
            """
            
            response = self.model.generate_content([prompt, enhanced_pil])
            response.resolve()
            
            return response.text.strip()
            
        except Exception as e:
            return f"Text extraction error: {str(e)}"

def main():
    st.title("Text Extraction from Turbulent Images")
    
    # Sidebar for API key
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E",
        type="password"
    )
    
    # Initialize extractor
    extractor = TextFocusedExtractor(api_key)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload a turbulent image with text", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader("Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Enhance Image"):
                with st.spinner("Enhancing image..."):
                    enhanced = extractor.enhance_text(image)
                    st.subheader("Enhanced Image")
                    st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        with col2:
            if st.button("Extract Text"):
                with st.spinner("Extracting text..."):
                    extracted_text = extractor.extract_text(image)
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
