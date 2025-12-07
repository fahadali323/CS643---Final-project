FROM python:3.10-slim

# Install Java 11 for PySpark compatibility
RUN apt-get update && \
    apt-get install -y openjdk-11-jre && \
    apt-get clean

# Install PySpark
RUN pip install pyspark==3.4.1

# Set working directory
WORKDIR /app

# Copy inference script
COPY predict2.py /app/predict2.py

# Optional: include model inside image (helps TAs run without mounting)
COPY wine_quality_model_final /app/wine_quality_model_final

# Allow script arguments:
# Usage inside container: predict2.py <model_path> <csv_path>
ENTRYPOINT ["python", "/app/predict2.py"]
