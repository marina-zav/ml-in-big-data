package org.apache.spark.ml.made

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.feature.{LSH, LSHModel}
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWriter, Identifiable, MLWriter, SchemaUtils}
import org.apache.spark.sql.types.StructType

import scala.util.Random

class RandLSH(override val uid: String) extends LSH[RandLSHModel] {
  private final val random = new Random(0)

  def this() = this(Identifiable.randomUID("rlsh"))

  override protected[this] def createRawLSHModel(inputDim: Int): RandLSHModel = {
    new RandLSHModel(uid, Array.fill($(numHashTables)) {
      linalg.Vectors.fromBreeze(breeze.linalg.Vector(Array.fill(inputDim)({
        if (random.nextInt() > 0) 1.0 else -1.0
      })))
    })
  }


  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
}

class RandLSHModel private[made](
                                  override val uid: String,
                                  private[made] val hyperplaneVectors: Array[Vector]
                                ) extends LSHModel[RandLSHModel] {

  override def setInputCol(value: String): this.type = super.set(inputCol, value)

  override def setOutputCol(value: String): this.type = super.set(outputCol, value)

  private[made] def this(hyperplaneVectors: Array[Vector]) = this(Identifiable.randomUID("rlsh"), hyperplaneVectors)

  override protected[ml] def hashFunction(vector: linalg.Vector): Array[linalg.Vector] = hyperplaneVectors
    .map(hyperplaneVector => if (vector.dot(hyperplaneVector) >= 0) 1 else -1)
    .map(Vectors.dense(_))

  override protected[ml] def keyDistance(x: linalg.Vector, y: linalg.Vector): Double = {
    val spaceDimension = 2
    if (Vectors.norm(x, spaceDimension) != 0 && Vectors.norm(y, spaceDimension) != 0) {
      1.0 - x.dot(y) / (Vectors.norm(x, spaceDimension) * Vectors.norm(y, spaceDimension))
    } else {
      1.0
    }
  }

  override protected[ml] def hashDistance(x: Seq[linalg.Vector], y: Seq[linalg.Vector]): Double =
    x.zip(y).map(item => if (item._1 == item._2) 1 else 0).sum.toDouble / x.size

  override def write: MLWriter = new RandLSHModelWriter(this)

  override def copy(extra: ParamMap): RandLSHModel = {
    val copied = new RandLSHModel(uid, hyperplaneVectors).setParent(parent)
    copyValues(copied, extra)
  }

  private[RandLSHModel] class RandLSHModelWriter(instance: RandLSHModel) extends MLWriter {

    private case class Data(hyperplaneVectors: Matrix)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(Matrices.dense(
        instance.hyperplaneVectors.length,
        instance.hyperplaneVectors.head.size,
        instance.hyperplaneVectors.map(_.toArray).reduce(Array.concat(_, _))
      ))
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }
}
