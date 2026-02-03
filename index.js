import { NeuralNetwork } from "./neuralnetwork.js";
///////////////////////MAIN AREA//////////////////////////////////
const canvas = document.getElementById('canvas');
const ctx = canvas?.getContext('2d');
if (!canvas || !ctx) {
    console.error("Canvas or context undefined");
    throw new Error("Canvas initialization failed");
}
// Create neural network first (needed by update function)
const nn = new NeuralNetwork(3, [4, 4], 2, 'tanh');
// Function to update canvas dimensions
function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    update();
}
// Set initial canvas dimensions
resizeCanvas();
// Update canvas dimensions when window resizes
window.addEventListener('resize', resizeCanvas);
// Start the animation loop
start();
setInterval(update, 10);
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
