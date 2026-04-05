import { ActivationFunction, NeuralNetwork, NeuralNetworkList } from "./neuralnetwork.js";
import { XGBoostEnsemble } from "./xgboost.js";

const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas?.getContext('2d');

if (!canvas || !ctx) {
    console.error("Canvas or context undefined");
    throw new Error("Canvas initialization failed");
}




let frame = 0;
window.addEventListener('resize', resizeCanvas);
window.onload = function() {
    // Reset all form elements to defaults so browser-cached values don't desync
    document.querySelectorAll('select').forEach(el => el.selectedIndex = 0);
    document.querySelectorAll('input[type="range"]').forEach(el => {
        (el as HTMLInputElement).value = (el as HTMLInputElement).defaultValue;
    });
    document.querySelectorAll('input[type="number"]').forEach(el => {
        (el as HTMLInputElement).value = (el as HTMLInputElement).defaultValue;
    });
    document.querySelectorAll('input[type="checkbox"]').forEach(el => {
        (el as HTMLInputElement).checked = (el as HTMLInputElement).defaultChecked;
    });
    resizeCanvas();
    start();
    setInterval(()=>{
        update();
        frame++;
    }, 1);
};

let isStarted = false;
let networkFormat: networkType = 'Val2in1out';
type networkType = 'Val2in1out' | 'Cat2in2out' | 'CatNout';
let numCategories = 3;
let showDataFormat: ShowDataType = 'output';
type ShowDataType = 'none' | 'output' | 'error' | 'test';
let showNetworkFormat: ShowNetworkType = 'best';
type ShowNetworkType = 'best' | 'all' | 'base' | 'latest';
let testFunctionVal2in1out: TestFunctionTypeVal2 = 'wave';
type TestFunctionTypeVal2 = 'wave' | 'radial' | 'xy' | 'checkerboard' | 'spiral' | 'diagonal';
let testFunctionCat2in2out: TestFunctionTypeCat2in2Out = 'circle';
type TestFunctionTypeCat2in2Out = 'circle' | 'square' | 'quadrants' | 'donut' | 'xor' | 'diagonal';
let testFunctionCatNout: TestFunctionTypeCatN = 'sectors';
type TestFunctionTypeCatN = 'sectors' | 'rings' | 'grid' | 'spiral';
let showTrainingData: ShowTrainingDataType = 'none';
type ShowTrainingDataType = 'none' | 'output' | 'error' | 'test';
let trainingMethod: TrainingMethod = 'genetic';
type TrainingMethod = 'genetic' | 'backprop' | 'XGBoost';
let input1 = 0;
let input2 = 0;
let generationsPerDrawCycle = 1;
let learningRate = 0.01; // Learning rate for backpropagation
let momentum = 0.9; // Momentum for backpropagation
let xgbMaxTrees = Infinity;
let xgbLimitTrees = false;
let xgbShrinkage = 0.1;
let xgbMaxDepth = 4;

// Colors for Cat2in2out visualization
let color1 = {r: 100, g: 150, b: 255}; // Light blue for class 1
let color2 = {r: 255, g: 100, b: 100}; // Light red for class 2

// Generate N distinct colors using HSL
function getCategoryColors(n: number): {r: number, g: number, b: number}[] {
    const colors: {r: number, g: number, b: number}[] = [];
    for (let i = 0; i < n; i++) {
        const hue = (i / n) * 360;
        const s = 0.7, l = 0.55;
        // HSL to RGB
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
        const m = l - c / 2;
        let r1 = 0, g1 = 0, b1 = 0;
        if (hue < 60) { r1 = c; g1 = x; }
        else if (hue < 120) { r1 = x; g1 = c; }
        else if (hue < 180) { g1 = c; b1 = x; }
        else if (hue < 240) { g1 = x; b1 = c; }
        else if (hue < 300) { r1 = x; b1 = c; }
        else { r1 = c; b1 = x; }
        colors.push({
            r: Math.round((r1 + m) * 255),
            g: Math.round((g1 + m) * 255),
            b: Math.round((b1 + m) * 255)
        });
    }
    return colors;
}

function drawNClassGrid(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number,
    axis1low: number, axis2low: number, axis1high: number, axis2high: number,
    rows: number, columns: number, showText: boolean,
    getOutputs: (in1: number, in2: number) => number[],
    colors: {r: number, g: number, b: number}[]) {
    const spaceX = width / columns;
    const spaceY = height / rows;
    ctx.save();
    ctx.fillStyle = '#e0e0e0';
    ctx.fillRect(left, top, width, height);
    for (let i = 0; i < columns; i++) {
        for (let j = 0; j < rows; j++) {
            const x = left + i * spaceX;
            const y = top + j * spaceY;
            const in1 = axis1low + i * (axis1high - axis1low) / (columns - 1);
            const in2 = axis2low + j * (axis2high - axis2low) / (rows - 1);
            const outputs = getOutputs(in1, in2);
            // Argmax to find winning class
            let maxIdx = 0;
            for (let k = 1; k < outputs.length; k++) {
                if (outputs[k] > outputs[maxIdx]) maxIdx = k;
            }
            const col = colors[maxIdx % colors.length];
            // Blend with confidence
            const confidence = Math.max(0, Math.min(1, outputs[maxIdx]));
            const r = Math.round(col.r * confidence + 240 * (1 - confidence));
            const g = Math.round(col.g * confidence + 240 * (1 - confidence));
            const b = Math.round(col.b * confidence + 240 * (1 - confidence));
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x, y, spaceX + (showText ? -1 : 1), spaceY + (showText ? -1 : 1));
            if (showText) {
                ctx.fillStyle = '#000';
                ctx.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(maxIdx.toString(), x + spaceX / 2, y + spaceY / 2);
            }
        }
    }
    ctx.restore();
}

///////////////////////MAIN AREA//////////////////////////////////
// Number of networks depends on training method: genetic needs 16, backprop needs 1
// Default is genetic (16 networks)
let numOfNeuralNetworks = 16;
let inputSize = 2;
let outputSize = 1;
let hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
let activationFunction: ActivationFunction = 'relu';
let outputActivationFunction: ActivationFunction = 'tanh';
let nnl: NeuralNetworkList | XGBoostEnsemble = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
let xgboost: XGBoostEnsemble | null = null;
let lastError = Infinity;
let topPanelScroll = 0;
let topPanelMaxScroll = 0;

canvas.addEventListener('wheel', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;
    const hch = canvas.height / 2;
    if (mouseY < hch && topPanelMaxScroll > 0) {
        e.preventDefault();
        // Normalize scroll delta across browsers/devices
        let delta = e.deltaY;
        if (e.deltaMode === 1) delta *= 30; // Line mode
        else if (e.deltaMode === 2) delta *= canvas.height; // Page mode
        topPanelScroll = Math.max(0, Math.min(topPanelMaxScroll, topPanelScroll + delta));
    }
}, { passive: false });

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
        let bestFitness = 0;

        if (trainingMethod === 'genetic') {
            // Genetic algorithm training
            if (nnl instanceof NeuralNetworkList) {
                let mutation_num_of_weights = 100;
                let mutation_weight_strength = 0.01;
                let mutation_num_of_biases = 100;
                let mutation_bias_strength = 0.01;

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
            }
        } else if (trainingMethod === 'backprop') {
            // Update learning rate and momentum for all networks
            if (nnl instanceof NeuralNetworkList) {
                nnl.setLearningRate(learningRate);
                nnl.setMomentum(momentum);

                for (let i = 0; i < generationsPerDrawCycle; i++){
                    bestFitness = nnl.trainBackpropagation(100);
                }
            }
        } else if (trainingMethod === 'XGBoost') {
            // XGBoost: decision tree ensemble trained on residuals
            if (xgboost && nnl instanceof NeuralNetworkList) {
                for (let i = 0; i < generationsPerDrawCycle; i++){
                    xgboost.train(nnl.trialInputsList, nnl.trialOutputsList, testInputsGrid, test);
                }
                bestFitness = xgboost.trainRMSE;
            }
        }

        // Compute test error for neural network methods
        if (trainingMethod !== 'XGBoost' && nnl instanceof NeuralNetworkList) {
            nnl.computeTestErr((inputs) => {
                const result = nnl instanceof NeuralNetworkList ? nnl.neuralNetworks[0].run(inputs) : {neurons: [{value: 0}]};
                return (result as any).neurons.map((n: any) => n.value);
            });
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
    const rowSize = 4;
    const xgbInput = [input1, input2];
    switch(showNetworkFormat){
        case "all":
            if (trainingMethod === 'XGBoost' && xgboost) {
                topPanelMaxScroll = Math.max(0, xgboost.getContentHeight(displayHeaderHeight, hch) - hch);
                topPanelScroll = Math.min(topPanelScroll, topPanelMaxScroll);
                xgboost.draw(ctx, 0, 0, canvas.width, hch, displayHeaderHeight, topPanelScroll, xgbInput);
            } else if (nnl instanceof NeuralNetworkList) {
                topPanelMaxScroll = Math.max(0, nnl.getContentHeight(rowSize, displayHeaderHeight, displayHeader, hch) - hch);
                topPanelScroll = Math.min(topPanelScroll, topPanelMaxScroll);
                nnl.draw(ctx, 0, 0, canvas.width, hch, rowSize, displayHeaderHeight, displayHeader, displayErrorDigits, displayMeanError, topPanelScroll);
            }
            break;
        case "base":
            if (xgboost) {
                topPanelMaxScroll = 0;
                topPanelScroll = 0;
                xgboost.drawSingleTree(ctx, 0, 0, canvas.width, hch, displayHeaderHeight, 0, xgbInput);
            }
            break;
        case "latest":
            if (xgboost) {
                topPanelMaxScroll = 0;
                topPanelScroll = 0;
                xgboost.drawSingleTree(ctx, 0, 0, canvas.width, hch, displayHeaderHeight, xgboost.trees.length - 1, xgbInput);
            }
            break;
        case "best":
            if (nnl instanceof NeuralNetworkList) {
                topPanelMaxScroll = 0;
                topPanelScroll = 0;
                nnl.drawHeader(ctx, 0, 0, canvas.width, displayHeaderHeight);
                let n = nnl.neuralNetworks[0].clone();
                n.run([input1, input2]);
                n.draw(ctx, 0, displayHeaderHeight, canvas.width, hch - displayHeaderHeight, displayErrorDigits, displayMeanError);
            }
            break;
    }




    const axis1low = -1;
    const axis1high = 1;
    const axis2low = -1;
    const axis2high = 1;


    const decimals = -1;
    const rows = 101;
    const columns = 101;

    const ouputMiddle = 0;
    const outputRange = 1;
    //V for visible (as in the 11x11 one cuz you can see the values) not vendetta
    const decimalsV = 2;
    const rowsV = 11;
    const columnsV = 11;



    // Prediction function wrapper (handles both neural networks and XGBoost)
    const predict = (inputs: number[]): number[] => {
        if (trainingMethod === 'XGBoost' && xgboost) {
            return xgboost.predict(inputs);
        } else if (nnl instanceof NeuralNetworkList) {
            const result = nnl.neuralNetworks[0].run(inputs);
            return result.neurons.map(n => n.value);
        }
        return [0];
    };

    // Display based on network format
    if (nnl instanceof NeuralNetworkList) {
        const displayNetwork = nnl.neuralNetworks[0];
        const isXGB = trainingMethod === 'XGBoost' && xgboost;

        if (networkFormat === 'Val2in1out') {
            switch(showDataFormat){
                case "none":
                    ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
                    if (isXGB) {
                        displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true,
                            (in1, in2) => predict([in1, in2])[0],
                            (val) => val * outputRange - ouputMiddle);
                    } else {
                        displayNetwork.display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV,  decimalsV, true, true);
                    }
                    break;
                case "error":
                    let errorRange = 1;
                    if (isXGB) {
                        const errorFn = (in1: number, in2: number) => predict([in1, in2])[0] - test([in1, in2])[0];
                        displayNetwork.displayGrid2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, errorFn, (val) => val * errorRange);
                        displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, errorFn, (val) => val * errorRange);
                    } else {
                        displayNetwork.display2Input1OutputError(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rows, columns, decimals, false, false, test);
                        displayNetwork.display2Input1OutputError(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, errorRange, rowsV, columnsV, decimalsV, true, true, test);
                    }
                    break;
                case "output":
                    if (isXGB) {
                        const outputFn = (in1: number, in2: number) => predict([in1, in2])[0];
                        const colorFn = (val: number) => val * outputRange - ouputMiddle;
                        displayNetwork.displayGrid2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, outputFn, colorFn);
                        displayNetwork.displayGrid2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, outputFn, colorFn);
                    } else {
                        displayNetwork.display2Input1Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false);
                        displayNetwork.display2Input1Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rowsV, columnsV,  decimalsV, true, true);
                    }
                    break;
                case "test":
                    displayNetwork.display2Input1OutputTest(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange, rows, columns, decimals, false, false, test);
                    displayNetwork.display2Input1OutputTest(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high,  ouputMiddle, outputRange, rowsV, columnsV, decimalsV, true, true, test);
                    break;
            }
        } else if (networkFormat === 'Cat2in2out') {
            switch(showDataFormat){
                case "none":
                    ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
                    if (isXGB) {
                        displayNetwork.displayGrid2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2,
                            (in1, in2) => predict([in1, in2]));
                    } else {
                        displayNetwork.display2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2);
                    }
                    break;
                case "error":
                    if (isXGB) {
                        const errorColor1 = {r: 255, g: 255, b: 255};
                        const errorColor2 = {r: 255, g: 0, b: 0};
                        const errorFn = (in1: number, in2: number) => {
                            let pred = predict([in1, in2]);
                            let expected = test([in1, in2]);
                            let error0 = Math.abs(pred[0] - expected[0]);
                            let error1 = Math.abs(pred[1] - expected[1]);
                            let totalError = (error0 + error1) / 2;
                            return [totalError, 0];
                        };
                        displayNetwork.displayGrid2Input2Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, errorColor1, errorColor2, errorFn);
                        displayNetwork.displayGrid2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, errorColor1, errorColor2, errorFn);
                    } else {
                        displayNetwork.display2Input2OutputError(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2, test);
                        displayNetwork.display2Input2OutputError(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2, test);
                    }
                    break;
                case "output":
                    if (isXGB) {
                        const outputFn = (in1: number, in2: number) => predict([in1, in2]);
                        displayNetwork.displayGrid2Input2Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2, outputFn);
                        displayNetwork.displayGrid2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2, outputFn);
                    } else {
                        displayNetwork.display2Input2Output(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2);
                        displayNetwork.display2Input2Output(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2);
                    }
                    break;
                case "test":
                    displayNetwork.display2Input2OutputTest(ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, false, false, color1, color2, test);
                    displayNetwork.display2Input2OutputTest(ctx, hcw + padding * canvas.width, hch, hcw, hch, axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, decimalsV, true, true, color1, color2, test);
                    break;
            }
        } else if (networkFormat === 'CatNout') {
            const catColors = getCategoryColors(numCategories);
            const outputFn = (in1: number, in2: number) => predict([in1, in2]);
            const testFn = (in1: number, in2: number) => test([in1, in2]);
            switch(showDataFormat){
                case "none":
                    ctx.fillRect(0, hch, hcw - padding * canvas.width, hch);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, outputFn, catColors);
                    break;
                case "output":
                    drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rows, columns, false, outputFn, catColors);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, outputFn, catColors);
                    break;
                case "test":
                    drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rows, columns, false, testFn, catColors);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, testFn, catColors);
                    break;
                case "error": {
                    const errFn = (in1: number, in2: number) => {
                        const pred = predict([in1, in2]);
                        const expected = test([in1, in2]);
                        let sum = 0;
                        for (let k = 0; k < pred.length; k++) sum += Math.abs(pred[k] - expected[k]);
                        const err = sum / pred.length;
                        // Return as single-class with error as confidence
                        return [err];
                    };
                    const errColors = [{r: 255, g: 0, b: 0}];
                    drawNClassGrid(ctx, 0, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rows, columns, false, errFn, errColors);
                    drawNClassGrid(ctx, hcw + padding * canvas.width, hch, hcw - padding * canvas.width, hch,
                        axis1low, axis2low, axis1high, axis2high, rowsV, columnsV, true, errFn, errColors);
                    break;
                }
            }
        }
    }
    // Show training data based on network format (only for neural networks)
    if (nnl instanceof NeuralNetworkList) {
        if (networkFormat === 'Val2in1out') {
            switch(showTrainingData){
                case "error":
                    nnl.display2Input1OutputDataPoints((inputs) => [predict(inputs)[0] - test(inputs)[0]], ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
                    break;
                case "output":
                    nnl.display2Input1OutputDataPoints((inputs) => [predict(inputs)[0]], ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
                    break;
                case "test":
                    nnl.display2Input1OutputDataPoints(test, ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
                    break;
            }
        } else if (networkFormat === 'Cat2in2out') {
            switch(showTrainingData){
                case "error":
                    // For error visualization, use white to red gradient
                    const errorColor1 = {r: 255, g: 255, b: 255}; // White for no error
                    const errorColor2 = {r: 255, g: 0, b: 0}; // Red for high error
                    nnl.display2Input2OutputDataPoints((inputs) => {
                        let nnOutput = predict(inputs);
                        let testOutput = test(inputs);
                        let error0 = Math.abs(nnOutput[0] - testOutput[0]);
                        let error1 = Math.abs(nnOutput[1] - testOutput[1]);
                        let totalError = (error0 + error1) / 2;
                        return [totalError, 0];
                    }, ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, errorColor1, errorColor2);
                    break;
                case "output":
                    nnl.display2Input2OutputDataPoints((inputs) => predict(inputs), ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, color1, color2);
                    break;
                case "test":
                    nnl.display2Input1OutputDataPoints(test, ctx, 0, hch, hcw - padding * canvas.width, hch, axis1low, axis2low, axis1high, axis2high, ouputMiddle, outputRange);
                    break;
            }
        }
        // CatNout uses the same grid display for training data overlay - no special data points needed
    }


}
let testInputsGrid: number[][] = [];

function createTrials(){
    const inputs = createInputs(inputSize, 1, -1, 0.1);
    // Test inputs: offset grid that doesn't overlap training data
    testInputsGrid = createInputs(inputSize, 0.95, -0.95, 0.1);
    if (nnl instanceof NeuralNetworkList) {
        nnl.createTrials(inputs, test);
        nnl.testInputs = testInputsGrid;
        nnl.testFn = test;
    }
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
        case "CatNout": {
            const n = numCategories;
            const oneHot = (idx: number) => {
                const arr = new Array(n).fill(0);
                arr[idx] = 1;
                return arr;
            };
            switch(testFunctionCatNout){
                case "sectors": {
                    let a = Math.atan2(inputs[1], inputs[0]); // -PI to PI
                    let sector = Math.floor(((a + Math.PI) / (2 * Math.PI)) * n);
                    return oneHot(Math.min(sector, n - 1));
                }
                case "rings": {
                    let dist = Math.sqrt(inputs[0] ** 2 + inputs[1] ** 2);
                    let ring = Math.floor(dist * n / 1.5);
                    return oneHot(Math.min(ring, n - 1));
                }
                case "grid": {
                    let gx = Math.floor((inputs[0] + 1) / 2 * Math.ceil(Math.sqrt(n)));
                    let gy = Math.floor((inputs[1] + 1) / 2 * Math.ceil(Math.sqrt(n)));
                    let idx = (gy * Math.ceil(Math.sqrt(n)) + gx) % n;
                    return oneHot(idx);
                }
                case "spiral": {
                    let sa = Math.atan2(inputs[1], inputs[0]);
                    let sr = Math.sqrt(inputs[0] ** 2 + inputs[1] ** 2);
                    let idx = Math.floor(((sa + Math.PI + sr * 4) % (2 * Math.PI)) / (2 * Math.PI) * n);
                    return oneHot(Math.min(Math.max(idx, 0), n - 1));
                }
            }
            break;
        }
    }
    return new Array(outputSize).fill(0);
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
            inputSize = 2;
            outputSize = 2;
            hiddenLayerSizes = [7, 10, 20, 20, 10, 7];
            activationFunction = 'relu';
            outputActivationFunction = 'sigmoid';

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
        case 'CatNout':
            inputSize = 2;
            outputSize = numCategories;
            hiddenLayerSizes = [10, 20, 20, 10];
            activationFunction = 'relu';
            outputActivationFunction = 'sigmoid';

            testFunctionDropdown.innerHTML = `
                <option value="sectors">Sectors</option>
                <option value="rings">Rings</option>
                <option value="grid">Grid</option>
                <option value="spiral">Spiral</option>
            `;
            testFunctionDropdown.value = testFunctionCatNout;
            break;
    }

    // Show/hide categories slider
    (document.getElementById('numCategoriesGroup') as HTMLDivElement).style.display =
        networkFormat === 'CatNout' ? '' : 'none';

    // Set number of networks based on training method
    if (trainingMethod === 'backprop') {
        numOfNeuralNetworks = 1;
    } else if (trainingMethod === 'XGBoost') {
        numOfNeuralNetworks = 1; // XGBoost starts with 1 and grows
    } else {
        numOfNeuralNetworks = 16;
    }

    nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
    if (trainingMethod === 'XGBoost') {
        xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees);
    }
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
    } else if (networkFormat === 'Cat2in2out') {
        testFunctionCat2in2out = value as TestFunctionTypeCat2in2Out;
    } else if (networkFormat === 'CatNout') {
        testFunctionCatNout = value as TestFunctionTypeCatN;
    }
    createTrials();
}
(window as any).testFunctionChange = testFunctionChange;

function numCategoriesChange(){
    const slider = document.getElementById('numCategoriesSlider') as HTMLInputElement;
    const input = document.getElementById('numCategoriesInput') as HTMLInputElement;
    const display = document.getElementById('numCategoriesDisplay') as HTMLSpanElement;
    // Sync slider and input
    if (document.activeElement === slider) {
        input.value = slider.value;
    } else {
        slider.value = input.value;
    }
    numCategories = parseInt(slider.value);
    display.textContent = numCategories.toString();
    if (networkFormat === 'CatNout') {
        networkChange();
    }
}
(window as any).numCategoriesChange = numCategoriesChange;
function inputChange(){
    generationsPerDrawCycle = parseFloat((document.getElementById('generationsPerDrawCycle') as HTMLInputElement).value);
    learningRate = parseFloat((document.getElementById('learningRate') as HTMLInputElement).value);
    momentum = parseFloat((document.getElementById('momentum') as HTMLInputElement).value);
    let i1 = parseFloat((document.getElementById('input1') as HTMLInputElement).value);
    input1 = Number(isNaN(i1)? 0: i1);
    let i2 = parseFloat((document.getElementById('input2') as HTMLInputElement).value);
    input2 = Number(isNaN(i2)? 0: i2);
    (document.getElementById('input1Slider') as HTMLInputElement).value = input1.toString();
    (document.getElementById('input2Slider') as HTMLInputElement).value = input2.toString();
    (document.getElementById('generationsPerDrawCycleSlider') as HTMLInputElement).value = generationsPerDrawCycle.toString();
    (document.getElementById('learningRateSlider') as HTMLInputElement).value = learningRate.toString();
    (document.getElementById('momentumSlider') as HTMLInputElement).value = momentum.toString();
    // Update display spans
    (document.getElementById('learningRateDisplay') as HTMLSpanElement).textContent = learningRate.toFixed(3);
    (document.getElementById('momentumDisplay') as HTMLSpanElement).textContent = momentum.toFixed(2);
}
(window as any).inputChange = inputChange;
function inputSliderChange(){
    generationsPerDrawCycle = parseFloat((document.getElementById('generationsPerDrawCycleSlider') as HTMLInputElement).value);
    learningRate = parseFloat((document.getElementById('learningRateSlider') as HTMLInputElement).value);
    momentum = parseFloat((document.getElementById('momentumSlider') as HTMLInputElement).value);
    input1 = parseFloat((document.getElementById('input1Slider') as HTMLInputElement).value);
    input2 = parseFloat((document.getElementById('input2Slider') as HTMLInputElement).value);
    (document.getElementById('input1') as HTMLInputElement).value = input1.toString();
    (document.getElementById('input2') as HTMLInputElement).value = input2.toString();
    (document.getElementById('generationsPerDrawCycle') as HTMLInputElement).value = generationsPerDrawCycle.toString();
    (document.getElementById('learningRate') as HTMLInputElement).value = learningRate.toString();
    (document.getElementById('momentum') as HTMLInputElement).value = momentum.toString();
    // Update display spans
    (document.getElementById('learningRateDisplay') as HTMLSpanElement).textContent = learningRate.toFixed(3);
    (document.getElementById('momentumDisplay') as HTMLSpanElement).textContent = momentum.toFixed(2);
}
(window as any).inputSliderChange = inputSliderChange;
function reset(){
    topPanelScroll = 0;
    if (trainingMethod === 'XGBoost') {
        nnl = new NeuralNetworkList(1, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees);
    } else {
        nnl = new NeuralNetworkList(numOfNeuralNetworks, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = null;
    }
    createTrials();
}
(window as any).reset = reset;
function showTrainingDataChange(){
    showTrainingData = (document.getElementById('showTrainingData') as HTMLSelectElement).value as ShowTrainingDataType;
}
(window as any).showTrainingDataChange = showTrainingDataChange;
function trainingMethodChange(){
    const newTrainingMethod = (document.getElementById('trainingMethod') as HTMLSelectElement).value as TrainingMethod;

    // Switching to or from XGBoost requires recreating the model
    if (newTrainingMethod === 'XGBoost' && trainingMethod !== 'XGBoost') {
        // Switching TO XGBoost: create XGBoost ensemble
        nnl = new NeuralNetworkList(1, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = new XGBoostEnsemble(inputSize, outputSize, xgbShrinkage, xgbMaxDepth, xgbMaxTrees);
        createTrials();
        numOfNeuralNetworks = 1;
    } else if (newTrainingMethod !== 'XGBoost' && trainingMethod === 'XGBoost') {
        // Switching FROM XGBoost: recreate neural network list
        const newRequired = newTrainingMethod === 'genetic' ? 16 : 1;
        nnl = new NeuralNetworkList(newRequired, inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
        xgboost = null;
        createTrials();
        numOfNeuralNetworks = newRequired;
    } else if (nnl instanceof NeuralNetworkList) {
        // Switching between genetic and backprop
        const getRequiredNetworks = (method: TrainingMethod) => {
            if (method === 'genetic') return 16;
            if (method === 'backprop') return 1;
            return 16;
        };

        const oldRequired = getRequiredNetworks(trainingMethod);
        const newRequired = getRequiredNetworks(newTrainingMethod);

        if (newRequired !== oldRequired) {
            if (newRequired > oldRequired) {
                // Need more networks: clone the best one
                nnl.resetError();
                nnl.testErrorTrials(nnl.trialInputsList, nnl.trialOutputsList, nnl.trialPower);
                nnl.sort();

                const bestNetwork = nnl.neuralNetworks[0].clone();
                const newNetworks = [];
                for (let i = 0; i < newRequired; i++) {
                    newNetworks.push(bestNetwork.clone());
                }
                nnl.neuralNetworks = newNetworks;
            } else {
                // Need fewer networks: keep only the best ones
                nnl.resetError();
                nnl.testErrorTrials(nnl.trialInputsList, nnl.trialOutputsList, nnl.trialPower);
                nnl.sort();
                nnl.neuralNetworks = nnl.neuralNetworks.slice(0, newRequired);
            }

            nnl.numOfNeuralNetworks = newRequired;
            numOfNeuralNetworks = newRequired;
        }
    }

    trainingMethod = newTrainingMethod;
    topPanelScroll = 0;
    updateSettingsVisibility();
}
(window as any).trainingMethodChange = trainingMethodChange;

function updateSettingsVisibility() {
    const backpropSettings = document.getElementById('backpropSettings') as HTMLDivElement;
    const xgboostSettings = document.getElementById('xgboostSettings') as HTMLDivElement;
    const viewDropdown = document.getElementById('showNetworkFormat') as HTMLSelectElement;
    const viewLabel = viewDropdown.parentElement?.querySelector('label');

    if (trainingMethod === 'XGBoost') {
        backpropSettings.style.display = 'none';
        xgboostSettings.style.display = '';
        if (viewLabel) viewLabel.textContent = 'Tree View';
        viewDropdown.innerHTML = `
            <option value="base">Base Tree</option>
            <option value="latest">Latest Tree</option>
            <option value="all">All Trees</option>
        `;
        viewDropdown.value = 'base';
        showNetworkFormat = 'base';
    } else {
        backpropSettings.style.display = trainingMethod === 'backprop' ? '' : 'none';
        xgboostSettings.style.display = 'none';
        if (viewLabel) viewLabel.textContent = 'Network View';
        viewDropdown.innerHTML = `
            <option value="best">Best</option>
            <option value="all">All</option>
        `;
        viewDropdown.value = showNetworkFormat === 'all' ? 'all' : 'best';
        if (showNetworkFormat !== 'all') showNetworkFormat = 'best';
    }
    topPanelScroll = 0;
}

function limitTreesToggleChange() {
    xgbLimitTrees = (document.getElementById('limitTreesToggle') as HTMLInputElement).checked;
    const sliderGroup = document.getElementById('maxTreesSliderGroup') as HTMLDivElement;
    const display = document.getElementById('maxTreesDisplay') as HTMLSpanElement;
    if (xgbLimitTrees) {
        sliderGroup.style.display = '';
        xgbMaxTrees = parseInt((document.getElementById('maxTreesSlider') as HTMLInputElement).value);
        display.textContent = xgbMaxTrees.toString();
    } else {
        sliderGroup.style.display = 'none';
        xgbMaxTrees = Infinity;
        display.textContent = '\u221E';
    }
    if (xgboost) xgboost.maxTrees = xgbMaxTrees;
}
(window as any).limitTreesToggleChange = limitTreesToggleChange;

function xgboostSliderChange() {
    if (xgbLimitTrees) {
        xgbMaxTrees = parseInt((document.getElementById('maxTreesSlider') as HTMLInputElement).value);
        (document.getElementById('maxTrees') as HTMLInputElement).value = xgbMaxTrees.toString();
        (document.getElementById('maxTreesDisplay') as HTMLSpanElement).textContent = xgbMaxTrees.toString();
    }
    xgbShrinkage = parseFloat((document.getElementById('shrinkageSlider') as HTMLInputElement).value);
    xgbMaxDepth = parseInt((document.getElementById('maxDepthSlider') as HTMLInputElement).value);
    (document.getElementById('shrinkage') as HTMLInputElement).value = xgbShrinkage.toString();
    (document.getElementById('maxDepth') as HTMLInputElement).value = xgbMaxDepth.toString();
    (document.getElementById('shrinkageDisplay') as HTMLSpanElement).textContent = xgbShrinkage.toFixed(2);
    (document.getElementById('maxDepthDisplay') as HTMLSpanElement).textContent = xgbMaxDepth.toString();
    if (xgboost) {
        xgboost.maxTrees = xgbMaxTrees;
        xgboost.shrinkage = xgbShrinkage;
        xgboost.maxDepth = xgbMaxDepth;
    }
}
(window as any).xgboostSliderChange = xgboostSliderChange;

function xgboostInputChange() {
    if (xgbLimitTrees) {
        xgbMaxTrees = parseInt((document.getElementById('maxTrees') as HTMLInputElement).value);
        (document.getElementById('maxTreesSlider') as HTMLInputElement).value = xgbMaxTrees.toString();
        (document.getElementById('maxTreesDisplay') as HTMLSpanElement).textContent = xgbMaxTrees.toString();
    }
    xgbShrinkage = parseFloat((document.getElementById('shrinkage') as HTMLInputElement).value);
    xgbMaxDepth = parseInt((document.getElementById('maxDepth') as HTMLInputElement).value);
    (document.getElementById('shrinkageSlider') as HTMLInputElement).value = xgbShrinkage.toString();
    (document.getElementById('maxDepthSlider') as HTMLInputElement).value = xgbMaxDepth.toString();
    (document.getElementById('shrinkageDisplay') as HTMLSpanElement).textContent = xgbShrinkage.toFixed(2);
    (document.getElementById('maxDepthDisplay') as HTMLSpanElement).textContent = xgbMaxDepth.toString();
    if (xgboost) {
        xgboost.maxTrees = xgbMaxTrees;
        xgboost.shrinkage = xgbShrinkage;
        xgboost.maxDepth = xgbMaxDepth;
    }
}
(window as any).xgboostInputChange = xgboostInputChange;
///////////////////////UI AREA////////////////////////////////////
