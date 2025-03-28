from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, regexp_replace

spark = SparkSession.builder \
    .appName("Average Salary by Gender") \
    .getOrCreate()

# JSON file into a DataFrame
df = spark.read.option("multiline", "true").json("workers.json")

df.printSchema()
df.show(truncate=False)

# Remove commas from the Salary column and convert it to a float
if "Salary" in df.columns:
    
    df = df.withColumn("Salary", regexp_replace(col("Salary"), ",", "").cast("float"))

    # avg. salary for each gender
    average_salary_df = df.groupBy("Gender").agg(avg("Salary").alias("Average Salary"))

    average_salary_df.show()
else:
    print("JSON file error")

spark.stop()
