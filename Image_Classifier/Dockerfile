# Use an official Python image as base
FROM python:3.9

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files
COPY requirements.txt /app/
COPY main.py model.py preprocess.py image_classifier.h5 /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
