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
    // Run evolution every 100ms instead of 10ms to see changes better
    setInterval(()=>{
        update();
        frame++;
    }, 1);
};

let isStarted = false;
let networkFormat: networkType = '2in1out';
type networkType = '2in1out' | '2in2out';
let showDataFormat: ShowDataType = 'output';
type ShowDataType = 'output' | 'error' | 'test';
let showNetworkFormat: ShowNetworkType = 'best';
type ShowNetworkType = 'best' | 'all';
let testFunction: TestFunctionType = 'wave';
type TestFunctionType = 'wave' | 'radial' | 'xy';
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
    const displayErrorDigits = 5;
    const displayMeanError = true;
    const displayHeaderHeight = 50;
    const displayHeader = true;
    switch(showNetworkFormat){
        case "all":
            const rowSize = 4;
            nnl.draw(ctx, 0, 0, canvas.width, canvas.height / 2, rowSize, displayHeaderHeight, displayHeader, displayErrorDigits, displayMeanError);
            break;
        case "best":
            nnl.drawHeader(ctx, 0, 0, canvas.width, displayHeaderHeight);

            let n = nnl.neuralNetworks[0].clone();
            n.run([input1, input2]);
            n.draw(ctx, 0, displayHeaderHeight, canvas.width, canvas.height / 2 - displayHeaderHeight, displayErrorDigits, displayMeanError)
    }




    const axis1low = 1;
    const axis1high = -1;
    const axis2low = 1;
    const axis2high = -1;


    const decimals = -1;
    const rows = 101;
    const columns = 101;

    const ouputMiddle = 0;
    const outputRange = 1;
    //V for visible (as in the 11x11 one cuz you can see the values) not vendetta
    const decimalsV = 2;
    const rowsV = 11;
    const columnsV = 11;

    const padding = 0.001;
    ctx.fillRect(0, canvas.height / 2 - padding * canvas.width, canvas.width, canvas.height / 2 + padding * canvas.width);
    
    switch(showDataFormat){
        case "error":
            let errorRange = 1;
            nnl.neuralNetworks[0].display2Input1OutputError(test, ctx, 0, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, errorRange, rows, columns, decimals, false, false);
            nnl.neuralNetworks[0].display2Input1OutputError(test, ctx, canvas.width / 2 + padding * canvas.width, canvas.height / 2, canvas.width / 2, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, errorRange, rowsV, columnsV, decimalsV, true, true);
            break;
        case "output":
            
            nnl.neuralNetworks[0].display2Input1Output(ctx, 0, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false);
            nnl.neuralNetworks[0].display2Input1Output(ctx, canvas.width / 2 + padding * canvas.width, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true);
            break;
        case "test":

            nnl.neuralNetworks[0].display2Input1OutputTest(test, ctx, 0, canvas.height / 2, canvas.width / 2 - padding * canvas.width, canvas.height / 2, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false);
            nnl.neuralNetworks[0].display2Input1OutputTest(test, ctx, canvas.width / 2 + padding * canvas.width, canvas.height / 2, canvas.width / 2, canvas.height / 2, axis1low, axis2low, axis1high, axis2high,  ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true);
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
    switch(testFunction){
        case "wave":
            let out = inputs[0] > Math.sin(inputs[1] * 2 * Math.PI)**1 ? 1: -1;
            return [out];
        case "radial":
            let out1 = Math.max(Math.min(1-2*(inputs[0] ** 2 + inputs[1] ** 2),1), -1);
            return [out1];
        case "xy":
            let out2 = inputs[0] * inputs[1];
            return [out2];
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

function showDataChange(){
    showDataFormat = (document.getElementById('showDataFormat') as HTMLSelectElement).value as ShowDataType;
}
(window as any).showDataChange = showDataChange;
function showNetworkChange(){
    showNetworkFormat = (document.getElementById('showNetworkFormat') as HTMLSelectElement).value as ShowNetworkType;
}
(window as any).showNetworkChange = showNetworkChange;
function testFunctionChange(){
    testFunction = (document.getElementById('testFunction') as HTMLSelectElement).value as TestFunctionType;
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
///////////////////////UI AREA////////////////////////////////////
