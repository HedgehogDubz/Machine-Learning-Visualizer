import { NeuralNetwork } from "./neuralnetwork.js";

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas?.getContext('2d');

if (!canvas || !ctx) {
    console.error("Canvas or context undefined");
    throw new Error("Canvas initialization failed");
}





window.addEventListener('resize', resizeCanvas);
window.onload = function() {
    resizeCanvas();
    start();
    setInterval(update, 10);
};


///////////////////////MAIN AREA//////////////////////////////////

const nn = new NeuralNetwork(3, [4, 4], 2, 'tanh');

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    update();
}
function start() {
    // Initial draw
    nn.run([1, -1, 1]);
    nn.draw(ctx, 0, 0, canvas.width, canvas.height);
}

function update() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mutate, run, and draw
    nn.mutate(1, 1, 1, 1);
    nn.run([1, -1, 1]);
    nn.draw(ctx, 0, 0, canvas.width, canvas.height);
}
///////////////////////MAIN AREA//////////////////////////////////

