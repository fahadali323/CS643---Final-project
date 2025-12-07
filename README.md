# 🍷 Wine Quality Prediction — Spark ML Project  
### Distributed Machine Learning with PySpark (Regression + Classification Metrics)

This project trains and evaluates a Wine Quality prediction model using **Apache Spark MLlib**, handling **custom CSV parsing**, **feature engineering**, **pipeline modeling**, and **evaluation metrics** including:

- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)  
- **R² (Coefficient of Determination)**  
- **F1 Score** (Good/Bad wine classification)

The model is saved as a **Spark PipelineModel** and can be used for distributed inference.

---

## 📁 Project Structure

```
/
├── train.py              # Training script (distributed)
├── predict2.py           # Prediction script (local or distributed)
├── run_train.sh          # Helper shell script to train on Spark cluster
├── run_predict.sh        # Helper script to run prediction
├── requirements.txt      # Python dependencies if running locally
├── README.md             # This documentation
└── wine_quality_model/   # Output model directory (generated after training)
```

---

# 🔧 Requirements

### Software Needed
| Component | Version |
|----------|---------|
| Python | 3.8+ |
| Apache Spark | 3.3+ |
| Hadoop (optional for HDFS) | 3.x |
| Java | 8 or 11 |

### Python Dependencies
```
pyspark
```

---

# 📊 Dataset Format

Your dataset uses semicolon-separated rows:

```
7.4;0.7;0.0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
```

The columns are:

1. fixed acidity  
2. volatile acidity  
3. citric acid  
4. residual sugar  
5. chlorides  
6. free sulfur dioxide  
7. total sulfur dioxide  
8. density  
9. pH  
10. sulphates  
11. alcohol  
12. quality (LABEL)

---

# 🏋️‍♂️ Training the Model

### Local training (single machine)

```bash
spark-submit train.py TrainingDataset.csv ValidationDataset.csv wine_quality_model
```

---

### Distributed training (Spark Standalone cluster)

```bash
/opt/spark/bin/spark-submit   --master spark://<MASTER-IP>:7077   train.py TrainingDataset.csv ValidationDataset.csv wine_quality_model
```

---

### HDFS Mode (optional)

Upload datasets:

```bash
hdfs dfs -put TrainingDataset.csv /wine/
hdfs dfs -put ValidationDataset.csv /wine/
```

Train:

```bash
spark-submit train.py hdfs:///wine/TrainingDataset.csv hdfs:///wine/ValidationDataset.csv wine_quality_model
```

---

# 📈 Training Outputs & Metrics

`train.py` prints:

### Regression Metrics
- RMSE  
- MAE  
- R²

### Classification Metric
Wine is labeled as:

- `good` if quality ≥ 6  
- `bad` otherwise  

You get:

- **F1 score**

---

# 🔮 Running Predictions

### Local prediction

```bash
spark-submit predict2.py wine_quality_model ValidationDataset.csv
```

### Distributed prediction

```bash
spark-submit   --master spark://<MASTER-IP>:7077   predict2.py wine_quality_model ValidationDataset.csv
```

### Example Output:

```
fixed acidity | alcohol | quality | prediction
------------------------------------------------------
7.4           | 9.4     | 5       | 5.01
7.8           | 9.8     | 5       | 5.20
...
```

---

# 📦 Model Output Format

After training, Spark creates:

```
wine_quality_model/
    ├── metadata/
    └── data/
```

This directory contains:

- `VectorAssembler`
- `LinearRegressionModel`
- Pipeline transformations

---

# 🚀 Running With Provided Scripts

### Train:

```bash
bash run_train.sh
```

### Predict:

```bash
bash run_predict.sh
```

---

# 🧪 How F1 Score is Computed

Inside `train.py`:

- Convert regression prediction → classification label  
- Compare with ground truth  
- Compute F1 score with:

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```

---
