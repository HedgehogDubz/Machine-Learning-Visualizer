import { ActivationFunction, NeuralNetwork, NeuralNetworkList } from "./neuralnetwork.js";

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas?.getContext('2d');

if (!canvas || !ctx) {
    console.error("Canvas or context undefined");
    throw new Error("Canvas initialization failed");
}




let frame = 0;
window.addEventListener('resize', resizeCanvas);
window.onload = function() {
    resizeCanvas();
    start();
    setInterval(()=>{
        update();
        frame++;
    }, 1);
};

let isStarted = false;
let networkFormat: networkType = 'Val2in1out';
type networkType = 'Val2in1out' | 'Cat2in2out';
let showDataFormat: ShowDataType = 'output';
type ShowDataType = 'none' | 'output' | 'error' | 'test';
let showNetworkFormat: ShowNetworkType = 'best';
type ShowNetworkType = 'best' | 'all';
let testFunctionVal2in1out: TestFunctionTypeVal2 = 'wave';
type TestFunctionTypeVal2 = 'wave' | 'radial' | 'xy' | 'checkerboard' | 'spiral' | 'diagonal';
let testFunctionCat2in2out: TestFunctionTypeCat2in2Out = 'circle';
type TestFunctionTypeCat2in2Out = 'circle' | 'square' | 'quadrants' | 'donut' | 'xor' | 'diagonal';
let showTrainingData: ShowTrainingDataType = 'none';
type ShowTrainingDataType = 'none' | 'output' | 'error' | 'test';
let input1 = 0;
let input2 = 0;
let generationsPerDrawCycle = 1;

///////////////////////MAIN AREA//////////////////////////////////
let numOfNeuralNetworks = 16;
let inputSize = 2;
let outputSize = 1;
let hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
let activationFunction: ActivationFunction = 'relu';
let outputActivationFunction: ActivationFunction = 'tanh';
let nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
let lastError = Infinity;

function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    update();
}

function start() {
   
    createTrials();
}
function update() {
    if (isStarted){
        let mutation_num_of_weights = 100;
        let mutation_weight_strength = 0.01;
        let mutation_num_of_biases = 100;
        let mutation_bias_strength = 0.01;

        let bestFitness = 0;

        for (let i = 0; i < generationsPerDrawCycle; i++){
            bestFitness = nnl.runGeneration(mutation_num_of_weights, mutation_weight_strength, mutation_num_of_biases, mutation_bias_strength);
            if(bestFitness === lastError){
                mutation_bias_strength /= 1.001;
                mutation_weight_strength /= 1.001;
            } else {
                mutation_bias_strength *= 1.001;
                mutation_weight_strength *= 1.001;
            }
        
        }
        lastError = bestFitness;
    }
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const padding = 0.001;

    const hcw = canvas.width / 2;
    const hch = canvas.height / 2;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, hch - padding * canvas.width, canvas.width, padding * canvas.width * 2);

    const displayErrorDigits = 5;
    const displayMeanError = true;
    const displayHeaderHeight = 50;
    const displayHeader = true;
    switch(showNetworkFormat){
        case "all":
            const rowSize = 4;
            nnl.draw(ctx, 0, 0, canvas.width, hch, rowSize, displayHeaderHeight, displayHeader, displayErrorDigits, displayMeanError);
            break;
        case "best":
            nnl.drawHeader(ctx, 0, 0, canvas.width, displayHeaderHeight);

            let n = nnl.neuralNetworks[0].clone();
            n.run([input1, input2]);
            n.draw(ctx, 0, displayHeaderHeight, canvas.width, hch - displayHeaderHeight, displayErrorDigits, displayMeanError)
    }




    const axis1low = -1;
    const axis1high = 1;
    const axis2low = -1;
    const axis2high = 1;


    const decimals = -1;
    const rows = 51;
    const columns = 51;

    const ouputMiddle = 0;
    const outputRange = 1;
    //V for visible (as in the 11x11 one cuz you can see the values) not vendetta
    const decimalsV = 2;
    const rowsV = 11;
    const columnsV = 11;


    
    switch(showDataFormat){
        case "none":
            ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
            nnl.neuralNetworks[0].display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV,  decimalsV, true, true);
            break;
        case "error":
            let errorRange = 1;
            nnl.neuralNetworks[0].display2Input1OutputError(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rows, columns, decimals, false, false, test);
            nnl.neuralNetworks[0].display2Input1OutputError(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rowsV, columnsV, decimalsV, true, true, test);
            break;
        case "output":
            nnl.neuralNetworks[0].display2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false);
            nnl.neuralNetworks[0].display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV,  decimalsV, true, true);
            break;
        case "test":
            nnl.neuralNetworks[0].display2Input1OutputTest(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false, test);
            nnl.neuralNetworks[0].display2Input1OutputTest(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high,  ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true, test);
            break;
    }
    switch(showTrainingData){
        case "error":
            nnl.display2Input1OutputDataPoints((inputs) => [nnl.neuralNetworks[0].run(inputs).neurons[0].value - test(inputs)[0]], ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
            break;
        case "output":
            nnl.display2Input1OutputDataPoints((inputs) => [nnl.neuralNetworks[0].run(inputs).neurons[0].value], ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
            break;
        case "test":
            nnl.display2Input1OutputDataPoints(test, ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
            break;
    }


}
function createTrials(){
    const inputs = createInputs(inputSize, 1, -1, 0.1);
    nnl.createTrials(inputs, test);
}
function createInputs(numOfInputs: number, high: number, low: number, spacing: number): number[][] {
    const possibleValues: number[] = [];
    for (let v = low; v <= high; v += spacing) {
        possibleValues.push(v);
    }

    const result: number[][] = [];

    function generateCombinations(current: number[]) {
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

function test(inputs: number[]): number[] {
    switch(networkFormat){
        case "Val2in1out":
            switch(testFunctionVal2in1out){
                case "wave":
                    return [inputs[0] > Math.sin(inputs[1] * 2 * Math.PI) ? 1 : -1];
                case "radial":
                    return [Math.max(Math.min(1 - 2 * (inputs[0] ** 2 + inputs[1] ** 2), 1), -1)];
                case "xy":
                    return [inputs[0] * inputs[1]];
                case "checkerboard":
                    return [Math.floor(inputs[0] * 4) % 2 === Math.floor(inputs[1] * 4) % 2 ? 1 : -1];
                case "spiral":
                    let angle = Math.atan2(inputs[1], inputs[0]);
                    let radius = Math.sqrt(inputs[0] ** 2 + inputs[1] ** 2);
                    return [Math.sin(angle * 3 + radius * 5) > 0 ? 1 : -1];
                case "diagonal":
                    return [inputs[0] + inputs[1] > 0 ? 1 : -1];
            }
            break;
        case "Cat2in2out":
            switch(testFunctionCat2in2out){
                case "circle":
                    return inputs[0] ** 2 + inputs[1] ** 2 < 0.5 ? [1, 0] : [0, 1];
                case "square":
                    return Math.abs(inputs[0]) < 0.5 && Math.abs(inputs[1]) < 0.5 ? [1, 0] : [0, 1];
                case "quadrants":
                    return inputs[0] * inputs[1] > 0 ? [1, 0] : [0, 1];
                case "donut":
                    let r = inputs[0] ** 2 + inputs[1] ** 2;
                    return r > 0.25 && r < 0.75 ? [1, 0] : [0, 1];
                case "xor":
                    return (inputs[0] > 0) !== (inputs[1] > 0) ? [1, 0] : [0, 1];
                case "diagonal":
                    return inputs[0] > inputs[1] ? [1, 0] : [0, 1];
            }
            break;
    }
    return [0];
}

///////////////////////MAIN AREA//////////////////////////////////
///////////////////////UI AREA////////////////////////////////////
function startToggle(){
    const startButton = document.getElementById('startButton') as HTMLButtonElement;
    const startToggleText = document.getElementById('startToggleText') as HTMLParagraphElement;
    if (isStarted){
        startToggleText.innerHTML = 'Start';
    }else {
        startToggleText.innerHTML = 'Stop';
    }
    isStarted = !isStarted;
}
(window as any).startToggle = startToggle;
function networkChange(){
    networkFormat = (document.getElementById('networkFormat') as HTMLSelectElement).value as networkType;

    // Update test function dropdown options
    const testFunctionDropdown = document.getElementById('testFunction') as HTMLSelectElement;
    testFunctionDropdown.innerHTML = '';

    switch (networkFormat){
        case 'Val2in1out':
            numOfNeuralNetworks = 16;
            inputSize = 2;
            outputSize = 1;
            hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
            activationFunction = 'relu';
            outputActivationFunction = 'tanh';

            // Add Val2in1out options
            testFunctionDropdown.innerHTML = `
                <option value="wave">Wave</option>
                <option value="radial">Radial</option>
                <option value="xy">XY Product</option>
                <option value="checkerboard">Checkerboard</option>
                <option value="spiral">Spiral</option>
                <option value="diagonal">Diagonal</option>
            `;
            testFunctionDropdown.value = testFunctionVal2in1out;
            break;
        case 'Cat2in2out':
            numOfNeuralNetworks = 16;
            inputSize = 2;
            outputSize = 2;
            hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
            activationFunction = 'relu';
            outputActivationFunction = 'tanh';

            // Add Cat2in2out options
            testFunctionDropdown.innerHTML = `
                <option value="circle">Circle</option>
                <option value="square">Square</option>
                <option value="quadrants">Quadrants (XY Sign)</option>
                <option value="donut">Donut</option>
                <option value="xor">XOR</option>
                <option value="diagonal">Diagonal</option>
            `;
            testFunctionDropdown.value = testFunctionCat2in2out;
            break;
    }

    nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
    createTrials();
}
(window as any).networkChange = networkChange;
function showDataChange(){
    showDataFormat = (document.getElementById('showDataFormat') as HTMLSelectElement).value as ShowDataType;
}
(window as any).showDataChange = showDataChange;
function showNetworkChange(){
    showNetworkFormat = (document.getElementById('showNetworkFormat') as HTMLSelectElement).value as ShowNetworkType;
}
(window as any).showNetworkChange = showNetworkChange;
function testFunctionChange(){
    const value = (document.getElementById('testFunction') as HTMLSelectElement).value;
    if (networkFormat === 'Val2in1out') {
        testFunctionVal2in1out = value as TestFunctionTypeVal2;
    } else {
        testFunctionCat2in2out = value as TestFunctionTypeCat2in2Out;
    }
    createTrials();
}
(window as any).testFunctionChange = testFunctionChange;
function inputChange(){
    generationsPerDrawCycle = parseFloat((document.getElementById('generationsPerDrawCycle') as HTMLInputElement).value);
    let i1 = parseFloat((document.getElementById('input1') as HTMLInputElement).value);
    input1 = Number(isNaN(i1)? 0: i1);
    let i2 = parseFloat((document.getElementById('input2') as HTMLInputElement).value);
    input2 = Number(isNaN(i2)? 0: i2);
    (document.getElementById('input1Slider') as HTMLInputElement).value = input1.toString();
    (document.getElementById('input2Slider') as HTMLInputElement).value = input2.toString();
    (document.getElementById('generationsPerDrawCycleSlider') as HTMLInputElement).value = generationsPerDrawCycle.toString();
}
(window as any).inputChange = inputChange;
function inputSliderChange(){
    generationsPerDrawCycle = parseFloat((document.getElementById('generationsPerDrawCycleSlider') as HTMLInputElement).value);
    input1 = parseFloat((document.getElementById('input1Slider') as HTMLInputElement).value);
    input2 = parseFloat((document.getElementById('input2Slider') as HTMLInputElement).value);
    (document.getElementById('input1') as HTMLInputElement).value = input1.toString();
    (document.getElementById('input2') as HTMLInputElement).value = input2.toString();
    (document.getElementById('generationsPerDrawCycle') as HTMLInputElement).value = generationsPerDrawCycle.toString();
}
(window as any).inputSliderChange = inputSliderChange;
function reset(){
    nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
    createTrials();
}
(window as any).reset = reset;
function showTrainingDataChange(){
    showTrainingData = (document.getElementById('showTrainingData') as HTMLSelectElement).value as ShowTrainingDataType;
}
(window as any).showTrainingDataChange = showTrainingDataChange;
///////////////////////UI AREA////////////////////////////////////
