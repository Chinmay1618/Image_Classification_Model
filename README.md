# Image_Classification_Model

Build and Deploy an Image Classification Model
📌 Image Classification Model
This project implements an image classification model using FastAPI and MobileNetV2 to classify images into different categories. The model is trained on the CIFAR-10 dataset and can be accessed via an API for predictions.

🚀 Features
✔️ Image classification using MobileNetV2
✔️ FastAPI for API-based inference
✔️ Dockerized for easy deployment
✔️ Supports image upload for real-time predictions

🛠 Project Structure
bash
Copy
Edit
Image_Classification_Model/
│── Image_Classifier/
│ ├── main.py # FastAPI application
│ ├── model.py # Model definition and training
│ ├── preprocess.py # Image preprocessing functions
│ ├── requirements.txt # Python dependencies
│ ├── image_classifier.h5 # Trained model weights
│ ├── Dockerfile # Containerization setup
│ ├── .gitignore # Ignore unnecessary files
│ ├── README.md # Project documentation

🔧 Installation & Setup
1️⃣ Clone the Repository
sh
Copy
Edit
git clone https://github.com/your-username/Image_Classification_Model.git
cd Image_Classification_Model/Image_Classifier

2️⃣ Install Dependencies
sh
Copy
Edit
pip install -r requirements.txt

3️⃣ Run the FastAPI Server
sh
Copy
Edit
uvicorn main:app --host 0.0.0.0 --port 8000

4️⃣ Test API Using Swagger UI
Once the server is running, open:
👉 http://127.0.0.1:8000/docs

Here, you can upload an image and get classification results.

🐳 Docker Setup
If you want to run the application inside a Docker container, use the following steps:

1️⃣ Build the Docker Image
sh
Copy
Edit
docker build -t image-classifier .

2️⃣ Run the Container
sh
Copy
Edit
docker run -p 8000:8000 image-classifier
Now, you can access the API at http://127.0.0.1:8000/docs

🎯 API Endpoints
Method Endpoint Description
POST /predict Upload an image and get a classification result
Example Response:

json
Copy
Edit
{
"class": "horse",
"confidence": 0.89
}
📌 Technologies Used
🔹 Python 3.9
🔹 FastAPI
🔹 TensorFlow / Keras
🔹 OpenCV
🔹 Docker
🔹 Uvicorn

👨‍💻 Contributors
Chinmay Sonsurkar

⚡ To-Do / Future Enhancements
✅ Improve accuracy with fine-tuning
✅ Add more datasets for better generalization
✅ Implement authentication for API access

