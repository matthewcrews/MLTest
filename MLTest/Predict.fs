module MLTest.Prediction

open Microsoft.ML
open Microsoft.ML.Data

//let context = MLContext()
//let burgerModel, schema = context.Model.Load("Model_Burger.zip")

[<CLIMutable>]
type WeatherInput = {
  Temperature : single
  Weather : string
}

[<CLIMutable>]
type BurgerPrediction = {
  [<ColumnName("Score")>]
  BurgerSales : single
}

[<CLIMutable>]
type PizzaPrediction = {
  [<ColumnName("Score")>]
  PizzaSales : single
}

[<CLIMutable>]
type TacoPrediction = {
  [<ColumnName("Score")>]
  TacoSales : single
}

let private score<'TInput, 'TOutput 
                           when 'TInput : not struct 
                           and 'TOutput : not struct
                           and 'TOutput : (new : unit -> 'TOutput) > 
                           (modelFile: string) 
                           (weatherObservation: 'TInput) =
  
  let context = MLContext()
  let model, schema = context.Model.Load(modelFile)
  let predictionEngine = context.Model.CreatePredictionEngine<'TInput, 'TOutput>(model)
  predictionEngine.Predict weatherObservation
  

let predict () =

  let weatherSample = {
    Temperature = 80.0f
    Weather = "Cloudy"
  }

  let burgerSalesPrediction = score<WeatherInput, BurgerPrediction> "Model_Burger.zip" weatherSample
  let pizzaSalesPrediction = score<WeatherInput, PizzaPrediction> "Model_Pizza.zip" weatherSample
  let tacoSalesPrediction = score<WeatherInput, TacoPrediction> "Model_Taco.zip" weatherSample

  printfn "%A" burgerSalesPrediction
  printfn "%A" pizzaSalesPrediction
  printfn "%A" tacoSalesPrediction

  ()