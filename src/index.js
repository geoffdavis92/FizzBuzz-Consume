import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import "./styles.css";

const inputShape = 10;

// Utility function provided - converts numbers to binary tensors
const numToBinTensor = num =>
  tf.tensor(
    num
      .toString(2)
      .padStart(inputShape, "0")
      .split("")
      .map(Number)
  );

const doFizzyPrediction = async () => {
  const fizzBuzzResult = [];
  const modelPath = "https://s3.amazonaws.com/ir_public/ai/fizzbuzz/fizzbuzz-model.json";

  /* Workshop Task 1: Load the Pre-trained Model
   * - Use tf.loadLayersModel() to load the model from modelPath
   * - Remember this is an async operation!
   * Docs: https://js.tensorflow.org/api/latest/#loadLayersModel
   */
  const model = null; // Your code here

  /* Workshop Task 2: Prepare the Input Data
   * - Create a tensor containing binary representations of numbers 1-100
   * - Use tf.tidy() to prevent memory leaks
   * - Use tf.stack() to combine multiple tensors
   * - Hint: [...Array(100).keys()] creates [0,1,2,...,99]
   * - Use the provided numToBinTensor helper function
   * Docs: https://js.tensorflow.org/api/latest/#tidy
   *       https://js.tensorflow.org/api/latest/#stack
   */
  const first100 = null; // Your code here

  /* Workshop Task 3: Make Predictions
   * - Use model.predict() on your input tensor
   * - This will return a tensor containing predictions for all numbers
   * Docs: https://js.tensorflow.org/api/latest/#tf.LayersModel.predict
   */
  const resultData = null; // Your code here

  /* Workshop Task 4: Process the Results
   * - Use unstack() to separate the batch predictions
   * - For each prediction:
   *   1. Convert tensor to array using dataSync()
   *   2. Find index of highest value using Math.max()
   *   3. Map index to ["number", "fizz", "buzz", "fizzbuzz"]
   * Docs: https://js.tensorflow.org/api/latest/#unstack
   *       https://js.tensorflow.org/api/latest/#tf.Tensor.dataSync
   */
  // Your code here
  // Should populate fizzBuzzResult array

  /* Workshop Task 5: Cleanup
   * - Dispose of any tensors you created
   * - At minimum: first100, resultData, and model
   * Docs: https://js.tensorflow.org/api/latest/#dispose
   */
  // Your code here

  return fizzBuzzResult.join(", ");
};

// React component code - no need to modify below this line
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
