package org.apache.spark.ml.made


import breeze.linalg.{sum, Vector => BreezeVector}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  setDefault(inputCol, "features")
  setDefault(outputCol, "target")

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String, val learningRate: Double, val numIterations: Int)
  extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable
    with MLWritable {

  def optimize(dataset: Dataset[_]): Vector = {
    var coefficients: Vector = Vectors.fromBreeze(BreezeVector.zeros(dataset.columns.length + 1))
    val nFeatures = coefficients.size
    val nSamples = dataset.count().asInstanceOf[Double]

    def gradient(features: Vector, target: Double, coefficients: Vector): Vector = {
      val loss = sum(features.asBreeze *:* coefficients.asBreeze) - target
      val gradient = features.copy.asBreeze * loss / nSamples
      Vectors.fromBreeze(gradient)
    }

    val gradientSum: Aggregator[Row, Vector, Vector] = new Aggregator[Row, Vector, Vector] {
      def zero: Vector = Vectors.zeros(nFeatures)

      def reduce(acc: Vector, x: Row): Vector = {
        val sampleGrad = gradient(x.getAs[Vector]($(inputCol)), x.getAs[Double]($(outputCol)), coefficients)
        Vectors.fromBreeze(acc.asBreeze + sampleGrad.asBreeze)
      }

      def merge(acc1: Vector, acc2: Vector): Vector = Vectors.fromBreeze(acc1.asBreeze + acc2.asBreeze)

      def finish(output: Vector): Vector = output

      override def bufferEncoder: Encoder[Vector] = ExpressionEncoder()

      override def outputEncoder: Encoder[Vector] = ExpressionEncoder()
    }

    for (_ <- 1 until numIterations) {
      val gradient = dataset.select(gradientSum.toColumn.as[Vector](ExpressionEncoder())).first().asBreeze
      val DenseCoefficients: BreezeVector[Double] = coefficients.asBreeze.toDenseVector
      DenseCoefficients += - learningRate * gradient
      coefficients = Vectors.fromBreeze(DenseCoefficients)
    }

    coefficients
  }

  def this() = this(Identifiable.randomUID("LinearRegression"), 1e-6, 100)

  def this(learningRate: Double, numIterations: Int) = this(
    Identifiable.randomUID("LinearRegression"),
    learningRate,
    numIterations)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val coefficients = optimize(dataset).toDense
    copyValues(new LinearRegressionModel(coefficients))
      .setParent(this)
  }


  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = {
    copyValues(new LinearRegression(learningRate, numIterations))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val coefficients: DenseVector)
  extends Model[LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(coefficients), extra)

  private[made] def this(coefficients: DenseVector) =
    this(Identifiable.randomUID("linearRegressionModel"), coefficients)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val bCoefficients = coefficients.asBreeze

    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x: Vector) => {
        sum(x.asBreeze *:* bCoefficients)
      })

    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      sqlContext.createDataFrame(coefficients.toArray.map(Tuple1.apply)).
        repartition(1).write.parquet(path + "/coeffs")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val coefficients : DenseVector =  vectors.select(vectors("_1").as[Vector]).first().toDense

      val model = new LinearRegressionModel(uid="LinearRegressionModel", coefficients = coefficients)
      metadata.getAndSetParams(model)
      model
    }
  }
}