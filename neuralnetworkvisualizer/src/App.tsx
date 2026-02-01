import { useEffect, useState, useRef } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { NeuralNetwork } from './neuralnetwork'
function App() {

  const [count, setCount] = useState(0)
  let nn = new NeuralNetwork(3,[5, 7, 7, 5], 3, 'sigmoid');
  console.log("ran");
  nn.write();

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  useEffect(()=>{
    const canvas = canvasRef.current;
    if (!canvas) return;              
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    nn.draw(ctx, 0, 0, canvas.width , canvas.height );
  })
  return (
    <>
      <div className="container">
          <h1>Neural Network Visualizer</h1>
          <canvas id="canvas" ref={canvasRef} width="800" height="600"></canvas>
      </div>
    </>
  )
}

export default App
