import { NeuralNetworkList } from "./neuralnetwork.js";
const canvas = document.getElementById('canvas');
const ctx = canvas?.getContext('2d');
if (!canvas || !ctx) {
    console.error("Canvas or context undefined");
    throw new Error("Canvas initialization failed");
}
let frame = 0;
window.addEventListener('resize', resizeCanvas);
window.onload = function () {
    resizeCanvas();
    start();
    // Run evolution every 100ms instead of 10ms to see changes better
    setInterval(() => {
        update();
        frame++;
    }, 1);
};
///////////////////////MAIN AREA//////////////////////////////////
const nnl = new NeuralNetworkList(16, 2, [7, 7, 10, 7, 7], 1, 'tanh');
function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    update();
}
function start() {
    // Create training data: all combinations of 2 inputs from -1 to 1 with spacing 0.1
    const inputs = createInputs(2, 1, -1, 0.02);
    nnl.createTrials(inputs, test);
}
function update() {
    let mutation_num_of_weights = 10;
    let mutation_weight_strength = 1;
    let mutation_num_of_biases = 1;
    let mutation_bias_strength = 0.1;
    const bestFitness = nnl.runGeneration(mutation_num_of_weights, mutation_weight_strength, mutation_num_of_biases, mutation_bias_strength);
    if (frame % 100 == 0) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        nnl.draw(ctx, 0, 0, canvas.width, canvas.height / 2, 4, 5, true);
        nnl.neuralNetworks[0].display2Input1Output(ctx, 0, canvas.height / 2, canvas.width, canvas.height / 2, -1, -1, 1, 1, 101, 101, 2);
    }
}
function createInputs(numOfInputs, high, low, spacing) {
    const possibleValues = [];
    for (let v = low; v <= high; v += spacing) {
        possibleValues.push(v);
    }
    const result = [];
    function generateCombinations(current) {
        if (current.length === numOfInputs) {
            result.push([...current]);
            return;
        }
        for (const value of possibleValues) {
            current.push(value);
            generateCombinations(current);
            current.pop();
        }
    }
    generateCombinations([]);
    return result;
}
function test(inputs) {
    let out = Math.cos(20 * inputs[0] * inputs[1]);
    return [out];
}
///////////////////////MAIN AREA//////////////////////////////////
