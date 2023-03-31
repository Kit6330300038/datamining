import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.sql import functions
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", "15g") \
    .config("spark.logConf", "true") \
    .appName('my-app') \
    .getOrCreate()

df = spark.read.csv('fueltype.csv',header=True)
df = df.dropna(subset='fuelType')
df = df.fillna('0')


df.show(5)
df.printSchema()
df = df.fillna('0')

indexers = [StringIndexer(inputCol=column, outputCol=column+"_n").fit(df) for column in list(
    set(df.columns)-set(['_c0']))]#

pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)

df_r = df_r.select(*(col(c).cast("int").alias(c) for c in df_r.columns))
df_fea = df_r.drop('_c0','brand','name','bodyType','year','transmission','power','fuelType')

df_fea.show(5)
df_fea.printSchema()   
                        
assembler = VectorAssembler(inputCols=['brand_n', 'name_n','bodyType_n','year_n',
                                       'transmission_n','power_n'], outputCol='features')
output = assembler.transform(df_fea)

final_data = output.select('features', 'fuelType_n')

final_data.show(5)

train_data,test_data = final_data.randomSplit([0.7,0.3])


model = DecisionTreeClassifier(labelCol='fuelType_n',featuresCol='features', maxDepth=10)

# Create ParamGrid for Cross Validation
dtparamGrid = (ParamGridBuilder()
             .addGrid(model.maxDepth, [7, 10, 13, 15, 20])
             .addGrid(model.maxBins, [20, 40, 60, 80, 100])
             .build())

# Evaluate model
dtevaluator = MulticlassClassificationEvaluator(labelCol= 'fuelType_n',metricName='f1')#

# Create 5-fold CrossValidator
dtcv = CrossValidator(estimator = model,
                      estimatorParamMaps = dtparamGrid,
                      evaluator = dtevaluator,
                      numFolds = 5)

dtc_model = dtcv.fit(train_data)

dtc_preds = dtc_model.transform(test_data)

pred_acc = dtevaluator.evaluate(dtc_preds)

train_acc = dtevaluator.evaluate(dtc_model.transform(train_data))

pred_table = dtc_preds.select("features", "prediction",'probability', "fuelType_n")
pred_table.show(10)
pred_table.sample(False, 0.1, seed=0).limit(10).show()
print('')
print('============================================')
print('')
print('')
print('F1-score train : ',train_acc )
print('')
print('F1-score test : ',pred_acc )
print('')
print('')
print('============================================')
print('')