ThisBuild / version := "0.1"

ThisBuild / scalaVersion := "2.12.7"

lazy val root = (project in file("."))
  .settings(
    name := "rand-hyper-lsh"
  )

val sparkVersion = "3.0.2"
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion withSources()

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources())
