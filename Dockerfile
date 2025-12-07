FROM python:3.10-slim

# Install Java (OpenJDK 17 works with PySpark 3.4.x)
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir pyspark==3.4.1

# Set workdir
WORKDIR /app

# Copy prediction script
COPY predict2.py /app/predict2.py

# (Optional) Copy model directory IF it exists locally in your repo
# Remove this line if your repo does NOT contain the folder!
COPY wine_quality_model_final /app/wine_quality_model_final

# Allow script arguments from `docker run`
ENTRYPOINT ["python", "predict2.py"]
