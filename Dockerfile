FROM python:3.10-slim

# Install Java (OpenJDK 21 is the only JRE available in Debian Trixie)
RUN apt-get update && \
    apt-get install -y openjdk-21-jre && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PySpark
RUN pip install --no-cache-dir pyspark==3.4.1

# Set workdir
WORKDIR /app

# Copy prediction script
COPY predict2.py /app/predict2.py

# Optional: include model folder (remove if not in repo)
COPY wine_quality_model_final /app/wine_quality_model_final

# Allow CMD arguments to pass through
ENTRYPOINT ["python", "predict2.py"]
