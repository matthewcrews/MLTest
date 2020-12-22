module MLTest.DataGeneration


open System
open MathNet.Numerics.Distributions
open MathNet.Numerics.Statistics
open Spectre.Console
open FileHelpers

let rng = System.Random(123)

type Day = Day of int
type Temperature = Temperature of float
type Weather =
  | Sunny
  | Cloudy
  | Rainy

type Food =
  | Burger
  | Pizza
  | Taco

[<CLIMutable>]
[<DelimitedRecord(",")>]
type Output = {
  Day : int
  Temperature : float
  Weather : string
  BurgerSales : float
  PizzaSales : float
  TacoSales : float
}

module Output =

  let create (Day day) (Temperature temperature) (weather: Weather) burgerSales pizzaSales tacoSales =

    {
      Day = day
      Temperature = temperature
      Weather = match weather with
                | Sunny -> "Sunny"
                | Cloudy -> "Cloudy"
                | Rainy -> "Rainy"
      BurgerSales = burgerSales
      PizzaSales = pizzaSales
      TacoSales = tacoSales
    }

let private weathers = 
  [
    0, Sunny
    1, Cloudy
    2, Rainy
  ] |> Map

let private weatherOffset =
  [
    (Burger, Sunny), -30.0
    (Burger, Cloudy), 0.0
    (Burger, Rainy), 30.0
    (Pizza, Sunny), 0.0
    (Pizza, Cloudy), 80.0
    (Pizza, Rainy), -80.0
    (Taco, Sunny), 40.0
    (Taco, Cloudy), -40.0
    (Taco, Rainy), 0.0
  ] |> Map

let private foodTemperatureCoefficients =
  [
    Burger, 3.5
    Pizza, 2.8
    Taco, 4.0
  ] |> Map

let private foodTemperatureIntercepts =
  [
    Burger, 337.5
    Pizza, 690.0
    Taco, 400.0
  ] |> Map

let private demandModel (rng: Random) (weather: Weather) (Temperature temperature) (food: Food) =
  let lambda = weatherOffset.[food, weather] + 
               foodTemperatureIntercepts.[food] + 
               temperature * foodTemperatureCoefficients.[food]
  Poisson.Sample (rng, lambda)
  |> float


let generate numberOfDays outputFile =

  let days =
    seq {1 .. numberOfDays}
    |> Seq.map (fun d -> 
      {| Day = Day d
         Temperature = Temperature (ContinuousUniform.Sample (rng, 60.0, 90.0) |> Math.Truncate)
         Weather = weathers.[DiscreteUniform.Sample (rng, 0, 2)]
      |} )
    |> Seq.map (fun d ->
      {| d with BurgerSales = demandModel rng d.Weather d.Temperature Burger
                PizzaSales = demandModel rng d.Weather d.Temperature Pizza
                TacoSales = demandModel rng d.Weather d.Temperature Taco
      |}
    )

  let outputData =
    days
    |> Seq.map (fun d -> Output.create d.Day d.Temperature d.Weather d.BurgerSales d.PizzaSales d.TacoSales)

  let engine = FileHelperEngine<Output>()
  engine.HeaderText <- engine.GetFileHeader()
  engine.WriteFile(outputFile, outputData)

  let burgerStats =
    days
    |> Seq.map (fun d -> d.BurgerSales)
    |> DescriptiveStatistics

  let pizzaStats =
    days
    |> Seq.map (fun d -> d.PizzaSales)
    |> DescriptiveStatistics

  let tacoStats =
    days
    |> Seq.map (fun d -> d.TacoSales)
    |> DescriptiveStatistics


  let table = Table()
  table.AddColumn("Food") |> ignore
  table.AddColumn("Mean") |> ignore
  table.AddColumn("Variance") |> ignore
  table.AddColumn("StdDev") |> ignore

  table.AddRow ([|"Burger"; $"%.2f{burgerStats.Mean}"; $"%.2f{burgerStats.Variance}"; $"%.2f{burgerStats.StandardDeviation}"|]) |> ignore
  table.AddRow ([|"Pizza"; $"%.2f{pizzaStats.Mean}"; $"%.2f{pizzaStats.Variance}"; $"%.2f{pizzaStats.StandardDeviation}"|]) |> ignore
  table.AddRow ([|"Taco"; $"%.2f{tacoStats.Mean}"; $"%.2f{tacoStats.Variance}"; $"%.2f{tacoStats.StandardDeviation}"|]) |> ignore

  AnsiConsole.Render(table)