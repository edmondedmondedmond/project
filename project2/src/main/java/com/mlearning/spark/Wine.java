package com.mlearning.spark;

import java.util.*;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.ml.*;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.*;

public class Wine {
    public static void main(String[] args) {
        // From Spark documentation RDD Java version
        // https://spark.apache.org/docs/3.0.0-preview/mllib-data-types.html
        SparkConf conf = new SparkConf().setAppName("WineQualityPrediction");
        JavaSparkContext sc = new JavaSparkContext(conf);

        String path_training = "s3://dsqualitywine/TrainingDataset.csv";
        String path_validation = "s3://dsqualitywine/ValidationDataset.csv";

        // Spark docs // Spark docs https://spark.apache.org/docs/latest/api/java/org/apache/spark/api/java/JavaRDD.html
        JavaRDD<String> data_training = sc.textFile(path_training);
        JavaRDD<String> data_validation = sc.textFile(path_validation);

        // Remove first line from Data Mining from docs
        // https://spark.apache.org/docs/2.1.1/mllib-evaluation-metrics.html
        final String header_training = data_training.first();
        JavaRDD<String> data1 = data_training.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                return !s.equalsIgnoreCase(header_training);
            }
        });

        final String header_validation = data_validation.first();
        JavaRDD<String> data2 = data_validation.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                return !s.equalsIgnoreCase(header_validation);
            }
        });

        // Create Training Label
        JavaRDD<LabeledPoint> parsedData1 = data1
                .map(new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(";");
                        return new LabeledPoint(Double.parseDouble(parts[11]),
                                Vectors.dense(Double.parseDouble(parts[0]),
                                        Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2]),
                                        Double.parseDouble(parts[3]),
                                        Double.parseDouble(parts[4]),
                                        Double.parseDouble(parts[5]),
                                        Double.parseDouble(parts[6]),
                                        Double.parseDouble(parts[7]),
                                        Double.parseDouble(parts[8]),
                                        Double.parseDouble(parts[9]),
                                        Double.parseDouble(parts[10])));
                    }
                });


        // For validation
        JavaRDD<LabeledPoint> parsedData2 = data2
                .map(new Function<String, LabeledPoint>() {
                    public LabeledPoint call(String line) throws Exception {
                        String[] parts = line.split(";");
                        return new LabeledPoint(Double.parseDouble(parts[11]),
                                Vectors.dense(Double.parseDouble(parts[0]),
                                        Double.parseDouble(parts[1]),
                                        Double.parseDouble(parts[2]),
                                        Double.parseDouble(parts[3]),
                                        Double.parseDouble(parts[4]),
                                        Double.parseDouble(parts[5]),
                                        Double.parseDouble(parts[6]),
                                        Double.parseDouble(parts[7]),
                                        Double.parseDouble(parts[8]),
                                        Double.parseDouble(parts[9]),
                                        Double.parseDouble(parts[10])));
                    }
                });


        // https://spark.apache.org/docs/3.0.0-preview/ml-classification-regression.html
        final LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(10)
                .run(parsedData1.rdd());
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = parsedData1.map( p -> {
            Double prediction = model.predict(p.features());
            return new Tuple2<>(prediction, p.label());
        });
        // Result metrics
        final MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        System.out.println();
        System.out.println("Logistic Regression Validation Accuracy: " + metrics.accuracy());
        System.out.println();


        double f_score = metrics.weightedFMeasure();
        System.out.println();
        System.out.println("Validation F Measure = " + f_score);
        System.out.println();



        model.save(sc.sc(), "s3://dsqualitywine/LogisticRegressionModel/");


        sc.stop();

    }
}
