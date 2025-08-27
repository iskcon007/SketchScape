🎨 SketchScape: Real vs. Sketch Image Detector
📌 Overview

SketchScape is a Streamlit-powered web application that uses a pre-trained MobileNetV2 deep learning model to classify uploaded images as either Real or Sketch.
The app also provides Grad-CAM visualizations to highlight important regions of the image, as well as image transformations (grayscale, sepia, edge detection, invert colors, etc.).

Users can upload images, view predictions, experiment with transformations, and download a CSV log of all predictions.

🚀 Features
🖼️ Real vs. Sketch Classification – upload an image and get instant results.
🔥 Grad-CAM Visualization – see which regions of the image influenced the prediction.
🎨 Image Transformations (for real images only):
Grayscale
Color enhancement
Sepia filter
Edge detection
Invert colors
📄 CSV Prediction Log – automatically save predictions for later use.
🐳 Docker Support – run the app inside a container.
🌐 Streamlit Cloud Deployment – share the app with a public link.

🛠 Tech Stack
Frontend/UI → Streamlit
Model → MobileNetV2 (TensorFlow/Keras)
Visualization → Matplotlib, OpenCV
Data Handling → Pandas, NumPy
Deployment → Docker, Streamlit Cloud
