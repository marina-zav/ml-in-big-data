package org.apache.spark.ml.made

import breeze.linalg.Vector
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.flatspec._
import org.scalatest.matchers._

class RandLSHTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val precision = 0.0001
  val hyperplaneVectors = Array(
    Vectors.dense(1, 1, 1, -1),
    Vectors.dense(1, -1, -1, 1),
    Vectors.dense(-1, 1, -1, 1)
  )

  "Model" should "test hash function" in {
    val rlshModel: RandLSHModel = new RandLSHModel(hyperplaneVectors)
      .setInputCol("ft")
      .setOutputCol("hash")
    val hashFunctionResult = rlshModel.hashFunction(linalg.Vectors.fromBreeze(Vector(6, 3, 5, 4)))

    hashFunctionResult.length should be(3)
    hashFunctionResult(0)(0) should be(1.0)
    hashFunctionResult(1)(0) should be(1.0)
    hashFunctionResult(2)(0) should be(-1.0)
  }

  "Model" should "test hash distance" in {
    val rlshModel: RandLSHModel = new RandLSHModel(hyperplaneVectors)
      .setInputCol("ft")
      .setOutputCol("hash")
    val hashDistanceResult = rlshModel.hashDistance(
      rlshModel.hashFunction(linalg.Vectors.fromBreeze(Vector(6, 3, 5, 4))),
      rlshModel.hashFunction(linalg.Vectors.fromBreeze(Vector(1, 4, 2, 3)))
    )
    hashDistanceResult should be(0.3333 +- precision)
  }

  "Model" should "test key distance" in {
    val rlshModel: RandLSHModel = new RandLSHModel(hyperplaneVectors)
      .setInputCol("ft")
      .setOutputCol("hash")
    val keyDistanceResult = rlshModel.keyDistance(
      linalg.Vectors.fromBreeze(Vector(6, 3, 5, 4)),
      linalg.Vectors.fromBreeze(Vector(1, 4, 2, 3))
    )
    keyDistanceResult should be(0.2125 +- precision)
  }
}
