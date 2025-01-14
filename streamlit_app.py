import streamlit as st
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import google.generativeai as genai
import PIL.Image
import io

class GentleTextExtractor:
    def __init__(self, api_key='AIzaSyB-VpIY25J2Mo13Q8h26Au5W218SHO6dPs'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def gentle_enhance(self, image):
        """Gentle enhancement for better text readability"""
        # Convert to float32
        img_float = image.astype(np.float32) / 255.0
        
        # Mild denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            (img_float * 255).astype(np.uint8),
            None,
            h=10,  # Reduced strength
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        ).astype(np.float32) / 255.0
        
        # Very gentle sharpening
        kernel_sharp = np.array([[-0.5,-0.5,-0.5],
                               [-0.5, 5,-0.5],
                               [-0.5,-0.5,-0.5]]) * 0.5
        
        if len(image.shape) == 3:
            sharpened = np.zeros_like(denoised)
            for i in range(3):
                sharpened[..., i] = cv2.filter2D(denoised[..., i], -1, kernel_sharp)
        else:
            sharpened = cv2.filter2D(denoised, -1, kernel_sharp)
        
        # Subtle contrast enhancement
        enhanced = cv2.convertScaleAbs((sharpened * 255).astype(np.uint8), alpha=1.1, beta=0)
        
        return enhanced
    
    def extract_text(self, image):
        """Extract text from image"""
        try:
            # Apply gentle enhancement
            enhanced = self.gentle_enhance(image)
            
            # Convert to PIL Image
            enhanced_pil = PIL.Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            original_pil = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            prompt = """
            Read and extract the text from this image.
            The text appears to be a quote or saying.
            Please return the exact text with:
            - Proper line breaks
            - Correct punctuation
            - Any attribution or source (like website names)
            
            Focus on accuracy rather than making corrections.
            If any part is unclear, mark it with [...].
            """
            
            # Try with both original and enhanced
            response_enhanced = self.model.generate_content([prompt, enhanced_pil])
            response_original = self.model.generate_content([prompt, original_pil])
            
            response_enhanced.resolve()
            response_original.resolve()
            
            # Return both results
            return {
                'enhanced': response_enhanced.text.strip(),
                'original': response_original.text.strip()
            }
            
        except Exception as e:
            return f"Text extraction error: {str(e)}"

def main():
    st.title("Gentle Text Extraction from Images")
    
    # Sidebar for API key
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value="AIzaSyB-VpIY25J2Mo13Q8h26Au5W218SHO6dPs",
        type="password"
    )
    
    # Initialize extractor
    extractor = GentleTextExtractor(api_key)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image with text", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader("Input Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if st.button("Extract Text"):
            with st.spinner("Processing..."):
                results = extractor.extract_text(image)
                
                if isinstance(results, dict):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("From Original")
                        st.text_area("Original Text", results['original'], height=200)
                    
                    with col2:
                        st.subheader("From Enhanced")
                        st.text_area("Enhanced Text", results['enhanced'], height=200)
                    
                    # Combined results download
                    combined_text = f"Original Image Text:\n{results['original']}\n\nEnhanced Image Text:\n{results['enhanced']}"
                    st.download_button(
                        "Download All Results",
                        data=combined_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(results)

if __name__ == "__main__":
    main()
