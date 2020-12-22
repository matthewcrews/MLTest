// Learn more about F# at http://fsharp.org

open System
open MLTest

[<EntryPoint>]
let main argv =
    printfn "Hello World from F#!"

    let trainDataFile = "WeatherTrainingData.csv"
    let evaluationDataFile = "WeatherEvaluationData.csv"

    // Generate data for testing
    DataGeneration.generate 100 trainDataFile
    DataGeneration.generate 30 evaluationDataFile

    // Train the models
    Fitting.train trainDataFile "."

    // Predict the Evaluation data
    Prediction.predict ()

    printfn "Press any key to close..."
    Console.ReadKey() |> ignore

    0 // return an integer exit code
