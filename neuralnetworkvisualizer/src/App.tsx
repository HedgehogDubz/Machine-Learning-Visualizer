import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { NeuralNetwork } from './neuralnetwork'
function App() {

  const [count, setCount] = useState(0)
  let nn = new NeuralNetwork(3,[5, 5, 5], 3, 'sigmoid');
  console.log("ran");
  nn.write();
  return (
    <>
      <div className="container">
          <h1>Neural Network Visualizer</h1>
          <canvas id="canvas" width="800" height="600"></canvas>
      </div>
    </>
  )
}

export default App
