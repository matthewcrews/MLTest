module MLTest.Fitting

open Microsoft.ML
open Microsoft.ML.Data

[<CLIMutable>]
type BurgerData = {
  [<LoadColumn(1)>] Temperature : single
  [<LoadColumn(2)>] Weather : string
  [<LoadColumn(3); ColumnName("Label")>] BurgerSales : single
}

[<CLIMutable>]
type PizzaData = {
  [<LoadColumn(1)>] Temperature : single
  [<LoadColumn(2)>] Weather : string
  [<LoadColumn(4); ColumnName("Label")>] PizzaSales : single
}

[<CLIMutable>]
type TacoData = {
  [<LoadColumn(1)>] Temperature : single
  [<LoadColumn(2)>] Weather : string
  [<LoadColumn(5); ColumnName("Label")>] TacoSales : single
}


let private trainModel<'Input> food inputFile (outputDir: string) =

  let context = MLContext()
  let dataView = context.Data.LoadFromTextFile<'Input> (inputFile, hasHeader = true, separatorChar = ',')
  let partitions = context.Data.TrainTestSplit(dataView, testFraction = 0.2)
  let pipeline = 
    EstimatorChain()
      .Append(context.Transforms.Categorical.OneHotEncoding("Weather"))
      .Append(context.Transforms.NormalizeMeanVariance("Temperature"))
      .Append(context.Transforms.Concatenate("Features", "Weather", "Temperature"))
      .Append(context.Regression.Trainers.LbfgsPoissonRegression())
  let model = partitions.TrainSet |> pipeline.Fit
  let metrics = partitions.TestSet |> model.Transform |> context.Regression.Evaluate

  // show the metrics
  printfn "Model metrics:"
  printfn "  RMSE:%f" metrics.RootMeanSquaredError
  printfn "  MSE: %f" metrics.MeanSquaredError
  printfn "  MAE: %f" metrics.MeanAbsoluteError

  let outputFile = $"{outputDir}/Model_{food}.zip"
  context.Model.Save (model, dataView.Schema, outputFile)


let train inputFile outputDir =

  trainModel<BurgerData> "Burger" inputFile outputDir
  trainModel<PizzaData> "Pizza" inputFile outputDir
  trainModel<TacoData> "Taco" inputFile outputDir
