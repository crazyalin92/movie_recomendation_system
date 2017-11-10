/**
  * Created by ALINA on 10.11.2017.
  */

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

object MovieLensALSDF {

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  case class Movie(movieId: Int, movieName: String, rating: Float)

  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    return Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]) {

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate();

    import sparkSession.implicits._

    // load ratings and movie titles
    val movieLensHomeDir = args(0)

    //Load my ratings
    val myRating = sparkSession.read.textFile(movieLensHomeDir + "personalRatings.txt")
      .map(parseRating)
      .toDF()

    //Load Ratings
    val ratings = sparkSession
      .read.textFile(movieLensHomeDir + "ratings.dat")
      .map(parseRating)
      .toDF()

    //Load Movies
    val moviesRDD = sparkSession
      .read.textFile(movieLensHomeDir + "movies.dat").map { line =>
      val fields = line.split("::")
      (fields(0).toInt, fields(1))
    }

    //show the DataFrames
    ratings.show(10)
    myRating.show(10)

    val numRatings = ratings.distinct().count()
    val numUsers = ratings.select("userId").distinct().count()
    val numMovies = moviesRDD.count()

    // Get movies dictionary
    val movies = moviesRDD.collect.toMap

    println("Got " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

    val ratingWithMyRats = ratings.union(myRating)

    // Split dataset into training and testing parts
    val Array(training, test) = ratingWithMyRats.randomSplit(Array(0.5, 0.5))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(3)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    //Get trained model
    val model = als.fit(training)

    //Evaluate Model Calculate RMSE
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)

    println(s"Root-mean-square error = $rmse")

    //Get My Predictions
    val myPredictions = model.transform(myRating).na.drop

    //Show your recomendations
    val myMovies = myPredictions.map(r => Movie(r.getInt(1), movies(r.getInt(1)), r.getFloat(2))).toDF
    myMovies.show(100)
  }
}
