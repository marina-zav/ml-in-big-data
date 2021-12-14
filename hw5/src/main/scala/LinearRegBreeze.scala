package org.mlinbd

import breeze.linalg._
import breeze.stats.mean
import breeze.numerics.pow
import scala.util.Random


class LinearRegression() {

  var weights: DenseVector[Double] = DenseVector.ones[Double](0)

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    X * weights
  }

  def mse_loss(y_true: DenseVector[Double], y_pred: DenseVector[Double]): Double = {
    mean(pow(y_true - y_pred, 2))
  }

  def grad_step(X: DenseMatrix[Double], y: DenseVector[Double], lr: Double): Unit = {
    val y_pred = predict(X)
    val grad = pinv(X) * (y_pred - y)
    weights -= lr * grad
  }

  def fit(X: DenseMatrix[Double], y: DenseVector[Double], lr: Double = 0.001, max_iter: Int = 100_000, debug: Boolean = true): Unit = {
    weights = DenseVector.ones[Double](X.cols)

    var cur_loss = Double.MaxValue
    var cur_err = Double.MaxValue
    for (i <- 0 to max_iter; if cur_loss >= cur_err) {
      cur_loss = cur_err
      grad_step(X, y, lr)
      val y_pred_new = predict(X)
      cur_err = mse_loss(y, y_pred_new)
      if (i % (max_iter / 10) == 0 && debug) {
        println(s"iteration: $i, loss: $cur_err, weight: $weights")
      }
    }
  }
}

object LinearRegBreeze {
  def main(args: Array[String]): Unit = {
    def generate_data(weight: DenseVector[Double], size: Int = 100): (DenseMatrix[Double], DenseVector[Double]) = {
      val rand = new Random
      val x_max = 100
      val noise_max = 5
      val X = DenseMatrix.fill[Double](size, weight.length)(rand.nextDouble() * x_max)
      val noise = DenseVector.fill[Double](size)(rand.nextDouble() * noise_max)
      val y = X * weight + noise
      (X, y)
    }

    val true_weights = DenseVector[Double](values = 1.5, 0.3, -0.7)
    val (x, y) = generate_data(weight = true_weights)
    val model = new LinearRegression()
    model.fit(x, y, 0.01, 1000, debug = false)

    val (x_val, y_val) = generate_data(weight = true_weights)
    val y_pred = model.predict(x_val)
    val mse = model.mse_loss(y_val, y_pred)

    println(s"MSE: $mse")
    println(s"true_weights: $true_weights")
    println(s"model weights: ${model.weights}")
  }
}