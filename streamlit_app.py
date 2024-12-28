import streamlit as st
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import google.generativeai as genai
import PIL.Image
import io

class TurbulenceRestorer:
    def __init__(self, api_key='AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E'):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def restore_image(self, turbulent_image, deblur_strength=1.0, noise_reduction=0.01):
        """Restore turbulent image with adjustable parameters"""
        # Convert to float32
        img_float = turbulent_image.astype(np.float32) / 255.0
        
        # Edge-preserving bilateral filter
        smoothed = cv2.bilateralFilter(
            (img_float * 255).astype(np.uint8),
            d=9,
            sigmaColor=75 * deblur_strength,
            sigmaSpace=75 * deblur_strength
        ).astype(np.float32) / 255.0
        
        # Deconvolution
        kernel = cv2.getGaussianKernel(5, 1.0 * deblur_strength)
        kernel = kernel * kernel.T
        
        # Wiener deconvolution
        psf_fft = np.fft.fft2(kernel, smoothed.shape[:2])
        psf_fft_conj = np.conj(psf_fft)
        
        if len(turbulent_image.shape) == 3:
            deconvolved = np.zeros_like(smoothed)
            for i in range(3):
                channel_fft = np.fft.fft2(smoothed[..., i])
                wiener = (psf_fft_conj / (np.abs(psf_fft)**2 + noise_reduction))
                deconvolved[..., i] = np.real(np.fft.ifft2(channel_fft * wiener))
        else:
            img_fft = np.fft.fft2(smoothed)
            wiener = (psf_fft_conj / (np.abs(psf_fft)**2 + noise_reduction))
            deconvolved = np.real(np.fft.ifft2(img_fft * wiener))
        
        # Apply sharpening
        kernel_sharp = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]]) * deblur_strength * 0.5
        
        if len(turbulent_image.shape) == 3:
            sharpened = np.zeros_like(deconvolved)
            for i in range(3):
                sharpened[..., i] = cv2.filter2D(deconvolved[..., i], -1, kernel_sharp)
        else:
            sharpened = cv2.filter2D(deconvolved, -1, kernel_sharp)
        
        # Final cleanup and enhance contrast
        enhanced = cv2.convertScaleAbs(sharpened * 255, alpha=1.1, beta=0)
        
        return enhanced

    def analyze_results(self, original_image, restored_image):
        """Analyze the restoration results using Gemini"""
        try:
            # Convert images to PIL format
            original_pil = PIL.Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            restored_pil = PIL.Image.fromarray(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
            
            prompt = """
            Compare these two images (original turbulent and restored):
            1. Assess the improvement in clarity and detail
            2. Evaluate text readability if text is present
            3. Note any artifacts or issues in the restoration
            4. Suggest potential improvements
            """
            
            # Analyze both images
            response = self.model.generate_content([prompt, original_pil, restored_pil])
            response.resolve()
            
            return response.text
        except Exception as e:
            return f"Analysis error: {str(e)}"

def main():
    st.title("Turbulence Image Restoration")
    
    # Sidebar for API key
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        value="AIzaSyCcMZPrzP5me7Rl4pmAc1Nn5vUDSan5Q6E",
        type="password"
    )
    
    # Initialize restorer
    restorer = TurbulenceRestorer(api_key)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload a turbulent image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        turbulent_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display original image
        st.subheader("Input Turbulent Image")
        st.image(cv2.cvtColor(turbulent_image, cv2.COLOR_BGR2RGB))
        
        # Restoration parameters
        st.sidebar.subheader("Restoration Parameters")
        deblur_strength = st.sidebar.slider("Deblur Strength", 0.5, 2.0, 1.0, 0.1)
        noise_reduction = st.sidebar.slider("Noise Reduction", 0.001, 0.1, 0.01, 0.001)
        
        if st.button("Restore Image"):
            with st.spinner("Restoring image..."):
                # Restore image
                restored = restorer.restore_image(
                    turbulent_image,
                    deblur_strength=deblur_strength,
                    noise_reduction=noise_reduction
                )
                
                # Display restored image
                st.subheader("Restored Image")
                st.image(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
                
                # Save option
                restored_bytes = cv2.imencode('.png', restored)[1].tobytes()
                st.download_button(
                    "Download Restored Image",
                    data=restored_bytes,
                    file_name="restored_image.png",
                    mime="image/png"
                )
                
                # Analysis option
                if st.button("Analyze Results"):
                    with st.spinner("Analyzing restoration..."):
                        analysis = restorer.analyze_results(turbulent_image, restored)
                        st.subheader("Analysis Results")
                        st.write(analysis)

if __name__ == "__main__":
    main()
