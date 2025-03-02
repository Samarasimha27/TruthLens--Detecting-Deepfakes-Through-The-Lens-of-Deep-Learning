TruthLens is an image-based deepfake detection system that leverages an ensemble of ResNet50, EfficientNet-B0, and MobileNet-V2 to identify manipulated images with high accuracy. The model extracts facial features using Dlib, applies data augmentation, and fine-tunes pre-trained CNN architectures for robust detection.

🔥 Key Features
✅ CNN Ensemble Model – Combines ResNet50, EfficientNet-B0, and MobileNet-V2 for improved accuracy.
✅ Face Cropping with Dlib – Efficient face extraction to focus on key facial features.
✅ Data Augmentation – Enhances model generalization with transformations.
✅ PyTorch Implementation – Fully optimized with GPU acceleration.
✅ Comprehensive Evaluation – Achieves 94.94% training accuracy and 92.90% validation accuracy with high precision and recall.

🚀 Project Pipeline
1️⃣ Dataset Preparation – Organizes real and fake images with structured directory formats.
2️⃣ Face Detection & Cropping – Uses Dlib for extracting faces from images.
3️⃣ Feature Extraction & Training – Fine-tunes ResNet50, EfficientNet-B0, and MobileNet-V2 with a fully connected layer for classification.
4️⃣ Evaluation & Metrics – Analyzes performance with F1 Score, Precision, Recall, and ROC AUC.

📊 Performance Metrics
Metric	Score
Train Accuracy	94.94%
Validation Accuracy	92.90%
Precision	94.89%
Recall	98.13%
F1 Score	96.48%
ROC AUC	96.34%
🛠 Tech Stack
Python, PyTorch, Dlib, OpenCV
ResNet50, EfficientNet-B0, MobileNet-V2
CUDA for GPU Acceleration


Access My Trained Model via - https://drive.google.com/file/d/1_FkmsMpKwPTgKbp4FZNXm12BPWel5Rrz/view?usp=drive_link
