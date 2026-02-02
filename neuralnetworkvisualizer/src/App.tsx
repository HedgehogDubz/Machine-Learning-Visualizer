import { useEffect, useRef } from "react";
import "./App.css";
import { NeuralNetwork } from "./neuralnetwork";

function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const canvasWrapRef = useRef<HTMLDivElement | null>(null);

  const nn = new NeuralNetwork(3, [5, 7, 7, 5], 3, "tanh");

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = canvasWrapRef.current;
    if (!canvas || !wrap) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    nn.mutate(1, 1, 1, 0)

    const resizeAndDraw = () => {
      const w = wrap.clientWidth;
      const h = wrap.clientHeight;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      ctx.clearRect(0, 0, w, h);

      nn.write();
      nn.run([-1, 0, 1]);

      nn.draw(ctx, 0, 0, w, h);
      
    };
    setInterval(function(){
          nn.mutate(1, 0.01, 1, 0.01)
          nn.run([-1, 0, 1]);
          nn.draw(ctx, 0, 0, wrap.clientWidth, wrap.clientHeight);
    }, 1)

    const ro = new ResizeObserver(resizeAndDraw);
    ro.observe(wrap);

    resizeAndDraw();

    return () => {
      ro.disconnect();
    };
  }, []);

  return (
    <div className="container">
      <h1>Neural Network Visualizer</h1>
      <div ref={canvasWrapRef} className="canvasWrap">
        <canvas id="canvas" ref={canvasRef} />
      </div>
    </div>
  );
}

export default App;
