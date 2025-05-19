# NEURAL-STYLE-TRANSFER
 
"COMPANY": CODTECH IT SOLUTIONS

"NAME": NISHRITHA VEMULA

"INTERN ID": CODF274

"DOMAIN": ARTIFICIAL INTELLIGENCE MARKUP LANGUAGE

"DURATION": 4 WEEKS

"MENTOR" : NEELA SANTOSH

---

## Description  
This project implements a Neural Style Transfer model using TensorFlow and Keras, wrapped in a Streamlit web app for easy interaction. The model transfers the artistic style of one image onto the content of another, generating a unique blend of both. It uses a pre-trained VGG19 network to extract content and style features, then optimizes a generated image by minimizing content and style losses.

## Features  
- **Pre-trained VGG19 model:** Extracts high-level features for content and style representation.  
- **Customizable weights:** Users can balance the influence of content and style during optimization.  
- **Gram matrix calculation:** Efficiently captures style information by comparing feature correlations.  
- **Configurable parameters:** Adjustable image size, learning rate, and iteration count for fine-tuning.  
- **Streamlit interface:** Simple and clean UI for uploading images, previewing results, and downloading output.

## Installation  
Clone the repository:  
```bash
git clone https://https://github.com/Nishritha-vemula/NEURAL-STYLE-TRANSFER.git
cd NEURAL-STYLE-TRANSFER
```  

Install dependencies:  
```bash
pip install -r requirements.txt
```  

## Usage  
Run the Streamlit app:  
```bash
streamlit run style_transfer_app.py
```  

Upload your content and style images through the web interface. Click the "Stylize Image" button and wait for the stylized image to be generated. Once done, you can preview and download the output image.

## Example  
- **Content image:** A photo or image whose structure you want to preserve.
- 
- **Style image:** An artistic painting or image whose style you want to apply.

![Image](https://github.com/user-attachments/assets/1c30c630-32b8-41bb-8115-4172861cceff)

The output image will maintain the content’s structure while adopting the style’s textures and colors.

## Results  
The model produces visually appealing stylized images combining content and style. Below is an example showcasing the input content, style images, and the final output.

![Image](https://github.com/user-attachments/assets/c6b5f277-c703-4289-b9c8-80dfe3edef5e)

## Future Improvements  
- Support for higher resolution images.  
- Additional loss functions to enhance stylization quality.  
- Real-time GPU acceleration for faster processing.  
- Deployment on cloud platforms for wider accessibility.  

---

Thank you to CODTECH IT SOLUTIONS and mentor Neela Santhosh for their support during this internship.
