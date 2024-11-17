import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import "./styles.css";
const inputShape = 10;

//#region Binary Conversion Guide
/*
numToBinTensor converts numbers to binary tensor format:
Example converting 5:
5                          // Input number
"101"                      // toString(2)
"0000000101"              // padStart(10, "0")
["0","0","0","0","0","0","0","1","0","1"] // split("")
[0,0,0,0,0,0,0,1,0,1]     // map(Number)
Final tensor shape: [10] (a 1D tensor with 10 values)

Why each step?
1. toString(2): Converts to binary string
2. padStart(10, "0"): Ensures 10-bit format for model input
3. split(""): Creates array of characters
4. map(Number): Converts strings to numbers for tensor
*/
//#endregion

// Convert numbers to Binary tensors
// const numToBinTensor = num =>
  // tf.tensor(
  //   num
  //     .toString(2)
  //     .padStart(inputShape, "0")
  //     .split("")
  //     .map(Number) // DO NOT USE parseInt here
  // );

const doFizzyPrediction = async () => {
  const fizzBuzzResult = [];
  // Hosted with ❤ by Infinite Red
  const modelPath =
    "https://s3.amazonaws.com/ir_public/ai/fizzbuzz/fizzbuzz-model.json";

  /* STEP 1: Load the Model
   * - We use tf.loadLayersModel to load a pre-trained model
   * - The model is hosted on S3 and contains our FizzBuzz logic
   * - This returns a Promise, so we use await
   */
  // const model = await tf.loadLayersModel(modelPath);

  /* STEP 2: Prepare Input Data
   * - We use tf.tidy to automatically clean up intermediate tensors
   * - Create array of numbers 1-100 using Array(100).keys()
   * - Map each number to binary using numToBinTensor
   * - Stack them all together into a batch using tf.stack
   */
  // const first100 = tf.tidy(() =>
  //   tf.stack([...Array(100).keys()].map(x => numToBinTensor(x + 1)))
  // );

  /* STEP 3: Make Predictions
   * - Use model.predict to run inference on our input
   * - This returns a tensor of predictions for all 100 numbers
   */
  // const resultData = await model.predict(first100);

  /* STEP 4: Process Results
   * - Unstack splits our batch back into individual predictions
   * - For each prediction, we:
   *   1. Convert tensor to regular array with dataSync()
   *   2. Find the highest probability prediction
   *   3. Map that to either number, fizz, buzz, or fizzbuzz
   */
  // resultData.unstack().forEach((result, index) => {
  //   const resultData = result.dataSync();
  //   const winner = resultData.indexOf(Math.max(...resultData));
  //   fizzBuzzResult.push([index + 1, "fizz", "buzz", "fizzbuzz"][winner]);
  //   result.dispose();
  // });

  /* STEP 5: Cleanup
   * - Always dispose of tensors when done
   * - This prevents memory leaks
   */
  // first100.dispose();
  // resultData.dispose();
  // model.dispose();

  return fizzBuzzResult.join(", ");
};

const App = props => {
  const [fizzBuzzResult, setFizzBuzzResult] = useState("loading model...");
  useEffect(() => {
    doFizzyPrediction().then(setFizzBuzzResult);
  }, []);

  return (
    <div className="App">
      <h1>FizzBuzz AI / ML</h1>
      <img src="/gantrobot.png" width="150" alt="gant robot" />
      <h3>
        <a href="http://gantlaborde.com/">By Gant Laborde</a>
      </h3>
      <h2>{fizzBuzzResult}</h2>
    </div>
  );
};

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
