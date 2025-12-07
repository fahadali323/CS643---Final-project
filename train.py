from pyspark.sql import SparkSession
from pyspark.sql.functions import split, regexp_replace, col, size, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import sys
import os

COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]

def load_custom_csv(spark, path):
    df = spark.read.text(path)
    df = df.withColumn("value", regexp_replace("value", "\"", ""))
    df = df.filter(col("value").rlike("^[0-9]"))
    df = df.withColumn("split", split(col("value"), ";"))
    df = df.filter(size(col("split")) == 12)

    df = df.select([
        col("split").getItem(i).cast("double").alias(COLUMNS[i])
        for i in range(12)
    ])

    return df.na.drop()

def compute_metrics(predictions):
    print("\n=== REGRESSION METRICS ===")

    evaluator = RegressionEvaluator(labelCol="quality", predictionCol="prediction")

    rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    mae  = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
    r2   = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

    print(f"RMSE: {rmse}")
    print(f"MAE : {mae}")
    print(f"R2  : {r2}")

    print("\n=== F1 SCORE (QUALITY → GOOD/BAD CLASSIFICATION) ===")

    classified = predictions.withColumn(
        "prediction_class",
        when(col("prediction") >= 6, 1.0).otherwise(0.0)  # <-- FIX: DOUBLE
    ).withColumn(
        "label_class",
        when(col("quality") >= 6, 1.0).otherwise(0.0)      # <-- FIX: DOUBLE
    )

    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label_class",
        predictionCol="prediction_class",
        metricName="f1"
    )

    f1 = f1_eval.evaluate(classified)
    print(f"F1 Score: {f1}")

    return rmse, mae, r2, f1

def main(train_csv, val_csv, model_dir):
    spark = SparkSession.builder.appName("WineQualityPipelineFinal").getOrCreate()

    print("\n=== LOADING DATASETS ===")
    train_df = load_custom_csv(spark, train_csv)
    val_df = load_custom_csv(spark, val_csv)

    print("Training rows:", train_df.count())
    print("Validation rows:", val_df.count())

    features = COLUMNS[:-1]

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    lr = LinearRegression(featuresCol="features", labelCol="quality")

    pipeline = Pipeline(stages=[assembler, lr])

    print("\n=== FITTING PIPELINE ===")
    model = pipeline.fit(train_df)

    print("\n=== SAVING PIPELINE MODEL ===")
    abs_path = os.path.abspath(model_dir)
    model.write().overwrite().save(abs_path)
    print(f"MODEL SAVED AT: {abs_path}")

    print("\n=== EVALUATING MODEL ON VALIDATION SET ===")
    val_pred = model.transform(val_df)
    compute_metrics(val_pred)

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: train4.py <train_csv> <val_csv> <model_dir>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])
