from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import split, regexp_replace, col, size, when
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
import sys

COLUMNS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]

def load_custom_csv(spark, path):
    df = spark.read.text(path)
    df = df.withColumn("value", regexp_replace("value", "\"", ""))
    df = df.filter(df.value.rlike("^[0-9]"))
    df = df.withColumn("split", split("value", ";"))
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

def main(model_dir, input_csv):
    spark = SparkSession.builder.appName("WineQualityPredict").getOrCreate()

    print("\n=== LOADING INPUT DATA ===")
    df = load_custom_csv(spark, input_csv)
    print("Rows:", df.count())

    print("\n=== LOADING MODEL ===")
    jvm = spark._jvm
    model = jvm.org.apache.spark.ml.PipelineModel.load(model_dir)

    print("\n=== RUNNING PREDICTIONS ===")
    jpred = model.transform(df._jdf)
    predictions = DataFrame(jpred, spark)

    predictions.show(20, truncate=False)

    print("\n=== EVALUATION METRICS ===")
    compute_metrics(predictions)

    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: predict2.py <model_dir> <input_csv>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
