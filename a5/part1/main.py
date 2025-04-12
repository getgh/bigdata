from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import OneVsRest

spark = SparkSession.builder.appName("IrisClassification").getOrCreate()

# Loading the iris dataset
df = spark.read.csv("iris.csv", header=True, inferSchema=True)

print("DataFrame Schema:")
df.printSchema()
print("\nFirst few rows:")
df.show(5)

# convert labels to numeric
indexer = StringIndexer(inputCol="species", outputCol="label")
df = indexer.fit(df).transform(df)

# Assembles
assembler = VectorAssembler(
    inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    outputCol="features"
)

data = assembler.transform(df).select("features", "label")

# data split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# base classifier for LinearSVC
svm = LinearSVC(maxIter=10, regParam=0.1)

# using OneVsRest strategy for multiclass classification
ovr = OneVsRest(classifier=svm)
ovr_model = ovr.fit(train_data)
predictions = ovr_model.transform(test_data)

# creating and training MLP classifier
input_features = len(assembler.getInputCols())
output_classes = df.select("label").distinct().count()
layers = [input_features, 5, output_classes]  # Input layer, hidden layer, output layer

mlp = MultilayerPerceptronClassifier(
    maxIter=100,
    layers=layers,
    blockSize=128,
    seed=42
)

mlp_model = mlp.fit(train_data)
mlp_predictions = mlp_model.transform(test_data)

# checking both algorithms
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
svm_accuracy = evaluator.evaluate(predictions)
mlp_accuracy = evaluator.evaluate(mlp_predictions)

print(f"LinearSVC (One-vs-Rest) Accuracy: {svm_accuracy:.4f}")
print(f"MLP Classifier Accuracy: {mlp_accuracy:.4f}")

# save results
predictions.select("prediction", "label") \
    .coalesce(1) \
    .write.csv("svm_results.csv", header=True, mode="overwrite")

mlp_predictions.select("prediction", "label") \
    .coalesce(1) \
    .write.csv("mlp_results.csv", header=True, mode="overwrite")

spark.stop()
