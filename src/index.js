import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";
import * as tf from "@tensorflow/tfjs";
import "./styles.css";
const inputShape = 10;

// Convert numbers to Binary tensors
const numToBinTensor = num =>
  tf.tensor(
    num
      .toString(2)
      .padStart(inputShape, "0")
      .split("")
      .map(Number) // DO NOT USE parseInt here
  );

const doFizzyPrediction = async () => {
  const fizzBuzzResult = [];
  // Hosted with ❤ by Infinite Red
  const modelPath =
    "https://s3.amazonaws.com/ir_public/ai/fizzbuzz/fizzbuzz-model.json";
  const model = await tf.loadLayersModel(modelPath);
  // STEP 1: create a batch of 100 consecutive numbers to predict
  const first100 = tf.tidy(() =>
    tf.stack([...Array(100).keys()].map(x => numToBinTensor(x + 1)))
  );

  // STEP 2: Run model on input numbers (1 to 100)
  const resultData = await model.predict(first100);

  // STEP 3: Build list of highest predicted results
  resultData.unstack().forEach((result, index) => {
    const resultData = result.dataSync(); // tensor to array
    // find Max prediction index
    const winner = resultData.indexOf(Math.max(...resultData));
    fizzBuzzResult.push([index + 1, "fizz", "buzz", "fizzbuzz"][winner]);
    result.dispose();
  });
  first100.dispose();
  resultData.dispose();
  model.dispose();
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
