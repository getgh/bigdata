from pyspark import SparkContext

sc = SparkContext("local", "Average Salary by Gender")

# load data from txt file
rdd = sc.textFile("workers.txt")

# comma removal
def safe_parse(line):
    try:
        fields = line.split("\t")
        salary = float(fields[3].replace(",", ""))
        return (fields[2], salary)
    except Exception as e:
        print(f"error at : {line}, Error: {e}")
        return None

parsed_rdd = rdd.map(safe_parse).filter(lambda x: x is not None)

# mapping the data to (gender, salary) pairs
gender_salary_rdd = parsed_rdd

# data to get (gender, (total_salary, count))
gender_salary_count_rdd = gender_salary_rdd.mapValues(lambda salary: (salary, 1)) \
                                           .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))

# avg. salary for each gender
average_salary_rdd = gender_salary_count_rdd.mapValues(lambda total_count: total_count[0] / total_count[1])

average_salaries = average_salary_rdd.collect()
for gender, avg_salary in average_salaries:
    print(f"Average salary for {gender}: {avg_salary}")

sc.stop()
