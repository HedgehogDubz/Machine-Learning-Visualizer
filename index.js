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
let isStarted = false;
let showFormat = 'output';
///////////////////////MAIN AREA//////////////////////////////////
const nnl = new NeuralNetworkList(16, 2, [7, 7, 10, 7, 7], 1, 'tanh');
function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    update();
}
function start() {
    const inputs = createInputs(2, 1, -1, 0.1);
    nnl.createTrials(inputs, test);
}
function update() {
    if (!isStarted) {
        return;
    }
    let mutation_num_of_weights = 3;
    let mutation_weight_strength = 0.1;
    let mutation_num_of_biases = 3;
    let mutation_bias_strength = 0.1;
    const bestFitness = nnl.runGeneration(mutation_num_of_weights, mutation_weight_strength, mutation_num_of_biases, mutation_bias_strength);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    nnl.draw(ctx, 0, 0, canvas.width, canvas.height / 2, 4, 5, true);
    let axis1low = 1;
    let axis1high = -1;
    let axis2low = 1;
    let axis2high = -1;
    let decimals = -1;
    let rows = 101;
    let columns = 101;
    //V for visible (as in the 11x11 one cuz you can see the values) not vendetta
    let decimalsV = 2;
    let rowsV = 11;
    let columnsV = 11;
    let padding = 0.001;
    ctx.fillRect(0, canvas.height / 2 - padding * canvas.width, canvas.width, canvas.height / 2 + padding * canvas.width);
    if (showFormat === 'error') {
        let errorRange = 1;
        nnl.neuralNetworks[0].display2Input1OutputError(test, ctx, 0, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, errorRange, rows, columns, decimals);
        nnl.neuralNetworks[0].display2Input1OutputError(test, ctx, canvas.width / 2 + padding * canvas.width, canvas.height / 2, canvas.width / 2, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, errorRange, rowsV, columnsV, decimalsV);
    }
    else if (showFormat === 'output') {
        let ouputMiddle = 0;
        let outputRange = 1;
        nnl.neuralNetworks[0].display2Input1Output(ctx, 0, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals);
        nnl.neuralNetworks[0].display2Input1Output(ctx, canvas.width / 2 + padding * canvas.width, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV, decimalsV);
    }
    else if (showFormat === 'test') {
        let ouputMiddle = 0;
        let outputRange = 1;
        nnl.neuralNetworks[0].display2Input1OutputTest(test, ctx, 0, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals);
        nnl.neuralNetworks[0].display2Input1OutputTest(test, ctx, canvas.width / 2 + padding * canvas.width, canvas.height / 2, canvas.width / 2, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV, decimalsV);
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
    let out = inputs[0] > Math.sin(inputs[1] * 2 * Math.PI) ** 1 ? 1 : -1;
    return [out];
}
///////////////////////MAIN AREA//////////////////////////////////
///////////////////////UI AREA////////////////////////////////////
function startToggle() {
    const startButton = document.getElementById('startButton');
    const startToggleText = document.getElementById('startToggleText');
    if (isStarted) {
        startToggleText.innerHTML = 'Start';
    }
    else {
        startToggleText.innerHTML = 'Stop';
    }
    isStarted = !isStarted;
}
window.startToggle = startToggle;
function showChange() {
    showFormat = document.getElementById('showFormat').value;
}
window.showChange = showChange;
///////////////////////UI AREA////////////////////////////////////
