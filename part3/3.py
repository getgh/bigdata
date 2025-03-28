from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace

spark = SparkSession.builder \
    .appName("Average Salary by Gender using SparkSQL") \
    .getOrCreate()

# JSON file into a DataFrame
df = spark.read.option("multiline", "true").json("workers.json")

# remove commas and convert
df = df.withColumn("Salary", regexp_replace(df["Salary"], ",", "").cast("float"))

df.createOrReplaceTempView("workers")

# SparkSQL to get the avg. salary for each gender
average_salary_df = spark.sql("""
    SELECT Gender, AVG(Salary) AS Average_Salary
    FROM workers
    GROUP BY Gender
""")

print("Average Salary by Gender using SparkSQL")
average_salary_df.show()

spark.stop()
