FROM bde2020/spark-java-template:3.1.1-hadoop3.2

MAINTAINER edmond <cp94@njit.edu>

ENV SPARK_APPLICATION_JAR_NAME WineQualityPrediction.jar
ENV SPARK_APPLICATION_MAIN_CLASS WinePrediction
ENV SPARK_APPLICATION_ARGS "s3://dsqualitywine/ValidationDataset.csv s3://dsqualitywine/LogisticRegressionModel/"