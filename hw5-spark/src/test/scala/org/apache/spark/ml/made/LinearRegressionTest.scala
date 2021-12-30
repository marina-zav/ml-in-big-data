package org.apache.spark.ml.made


import breeze.linalg.DenseVector
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val train_data: DataFrame = LinearRegressionTest._train_data
  val delta: Double = 0.02
  val learningRate: Double = 0.1
  val numIterations: Int = 100

  val coefficients: Vector = Vectors.dense(2, 0.5, -0.6)

  "Model" should "return linear combination of the features" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      coefficients = coefficients.toDense
    ).setInputCol("features")
      .setOutputCol("target")

    val values = model.transform(data).collect().map(_.getAs[Double](1))

    values.length should be(data.count())

    values(0) should be(vectors(0)(0) * coefficients(0) + vectors(0)(1) * coefficients(1) + vectors(0)(2) * coefficients(2) +- delta)
    values(1) should be(vectors(1)(0) * coefficients(0) + vectors(1)(1) * coefficients(1) + vectors(1)(2) * coefficients(2) +- delta)
  }
}

object LinearRegressionTest extends WithSpark {

  import sqlc.implicits._

  lazy val _vectors = Seq(
    Vectors.dense(-2, -0.6, 1.6),
    Vectors.dense(-1.4, 0.3, 1.8),
  )
  lazy val _data: DataFrame = {
    _vectors.map(x => Tuple1(x)).toDF("features")
  }
  lazy val _train_points: Seq[Vector] = Seq.fill(100)(Vectors.fromBreeze(DenseVector.rand(3)))
  lazy val _train_data: DataFrame = {
    _train_points.map(x => Tuple1(x)).toDF("features")
  }
}