// sbt 1.5.6 (Oracle Corporation Java 13.0.1)

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.7"

lazy val root = (project in file("."))
  .settings(
    name := "lr-breeze",
    idePackagePrefix := Some("org.mlinbd")
  )

libraryDependencies += "org.scalanlp" %% "breeze" % "2.0"
