import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions
from pyspark.sql.functions import col
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import *


spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "15g") \
    .config("spark.logConf", "true") \
    .appName('my-app') \
    .getOrCreate()

df = spark.read.csv('fueltype.csv',header=True)
df = df.dropna(subset='fuelType')
df_g = df.select('*').where(df.fuelType == "Gasoline")
df_g.show(5)

df_d = df.select('*').where(df.fuelType == "Diesel")
df_d.show(5)

df_e = df.select('*').where(df.fuelType == "Electro")
df_e.show(5)

df_g_merged = df_g.select(array('brand','name','bodyType','year',
                                'transmission','power','fuelType').alias('item'))
df_g_merged = df_g_merged.withColumn('items', functions.expr('filter(item, x -> x is not null)'))

df_d_merged = df_d.select(array('brand','name','bodyType','year',
                                'transmission','power','fuelType').alias('item'))
df_d_merged = df_d_merged.withColumn('items', functions.expr('filter(item, x -> x is not null)'))

df_e_merged = df_e.select(array('brand','name','bodyType','year',
                                'transmission','power','fuelType').alias('item'))
df_e_merged = df_e_merged.withColumn('items', functions.expr('filter(item, x -> x is not null)'))


df_g_merged.show(5)
df_d_merged.show(5)
df_e_merged.show(5)

fp = FPGrowth(itemsCol="items", minSupport=0.2, minConfidence=0.3)
fpmodel1 = fp.fit(df_g_merged)
fpmodel2 = fp.fit(df_d_merged)
fpmodel3 = fp.fit(df_e_merged)

a1 = fpmodel1.associationRules.filter(array_contains(col("consequent"), "Gasoline"))
a1.select("antecedent","consequent","support","confidence").orderBy(
    col("support").desc(),col("confidence").desc()).show()
a2 = fpmodel2.associationRules.filter(array_contains(col("consequent"), "Diesel"))
a2.select("antecedent","consequent","support","confidence").orderBy(
    col("support").desc(),col("confidence").desc()).show()
a3 = fpmodel3.associationRules.filter(array_contains(col("consequent"), "Electro"))
a3.select("antecedent","consequent","support","confidence").orderBy(
    col("support").desc(),col("confidence").desc()).show()