import { drawCircle, Point } from './graphics.js';
////////////////////////////////////////Neural Network List///////////////////////////////////////
export class NeuralNetworkList {
    numOfNeuralNetworks = 0;
    neuralNetworks: NeuralNetwork[] = [];
    generation: number = 0;
    inputSize: number;
    outputSize: number;
    

    trialInputsList: number[][] = [];
    trialOutputsList: number[][] = [];
    trialPower: number = 2;
    trainingData = [];
    constructor(numOfNeuralNetworks:number, inputSize: number, hiddenLayerSizes: number[], outputSize: number, activationFunction: ActivationFunction, outputActivationFunction:ActivationFunction){
        this.numOfNeuralNetworks = numOfNeuralNetworks;
        for(let i = 0; i < numOfNeuralNetworks; i++){
            const nn = new NeuralNetwork(inputSize, hiddenLayerSizes, outputSize, activationFunction, outputActivationFunction);
            this.neuralNetworks.push(nn);
        }
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        
    }
    public run(inputs: number []){
        this.neuralNetworks.forEach(nn => nn.run(inputs));
    }
    public mutate(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number){
        this.neuralNetworks.forEach(nn => nn.mutate(numOfWeights,weightStrength, numOfBiases, biasesStrength));
        return this;
    }
    public draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, rowSize: number, displayHeaderHeight?:number, displayHeader?:boolean, displayErrorDigits?: number, displayMeanError?: boolean){
        let xSpace = width / rowSize;
        let ySpace = (height - (displayHeader ? displayHeaderHeight : 0)) / Math.ceil(this.neuralNetworks.length / rowSize);
        let col = 0;
        let row = 0;

        //draw header
        if (displayHeader){
            this.drawHeader(ctx, left, top, width, (displayHeader ? displayHeaderHeight : 0));
        }


        //draw networks
        this.neuralNetworks.forEach((nn: NeuralNetwork) => {
            nn.draw(ctx, left + xSpace * col, top + ySpace * row + (displayHeader? displayHeaderHeight: 0), xSpace, ySpace, displayErrorDigits, displayMeanError);
            col++;
            if(col >= rowSize){
                col = 0;
                row++;
            }
        });
    }
    public drawBest(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, rowSize: number, displayHeaderHeight?:number, displayHeader?:boolean, displayErrorDigits?: number, displayMeanError?: boolean){
        this.sort();
        //draw header
        if (displayHeader){
            this.drawHeader(ctx, left, top, width, displayHeaderHeight);
        }

        this.neuralNetworks[0].draw(ctx, left, top + (displayHeader? 20: 0), width, height - (displayHeader? 20: 0), displayErrorDigits, displayMeanError);
    }
    public drawHeader(ctx, left, top , width, height){
        ctx.save();
            ctx.fillStyle = "#e0e0e0"
            ctx.fillRect(left, top, width, height);
            let textStr = "Generation: " + this.generation;
            ctx.font = '12px sans-serif';
            ctx.textBaseline = 'middle';
            let measure = ctx.measureText(textStr);
            let w = measure.width;

            if(w >= width - 10) {
                textStr = "Gen: " + this.generation
                ctx.font = '12px sans-serif';
                measure = ctx.measureText(textStr);
                w = measure.width;

                if(w >= width - 10) {
                    textStr = textStr.slice(0, -1);
                }
            }
            ctx.fillStyle = "#000000"
            ctx.fillText(textStr, left + 5, top + height / 2);

            ctx.restore();

    }
    public testError(outputs: number[], power?:number): number{

        if(outputs.length !== this.outputSize){
            console.error("Wrong number of outputs, expected: " + this.outputSize + " | received: " + outputs.length)
            return 0
        }
        this.neuralNetworks.forEach((nn: NeuralNetwork) => {
            nn.testError(outputs, false, power);
        });

    }
    public display2Input1OutputDataPoints(test: (inputs: number[]) => number[], ctx, left, top, width, height, axis1low: number, axis2low: number, axis1high: number, axis2high: number, outputMiddle:number, outputRange: number){

        ctx.save();
        ctx.rect(left, top, width, height);
        ctx.clip();

        this.trialInputsList.forEach((input) => {
            const x = left + ((input[0] - axis1low) / (axis1high - axis1low)) * width;
            const y = top + ((input[1] - axis2low) / (axis2high - axis2low)) * height;

            ctx.fillStyle = "black";
            drawCircle(ctx, {x, y}, Math.min(width, height) / 100);
            ctx.fillStyle = this.numToColorWhite((test(input)[0] - outputMiddle) / outputRange);
            drawCircle(ctx, {x, y}, Math.min(width, height) / 120);
        })

        ctx.restore();

    }

    public display2Input2OutputDataPoints(test: (inputs: number[]) => number[], ctx, left, top, width, height, axis1low: number, axis2low: number, axis1high: number, axis2high: number, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}){

        ctx.save();
        ctx.rect(left, top, width, height);
        ctx.clip();

        this.trialInputsList.forEach((input) => {
            const x = left + ((input[0] - axis1low) / (axis1high - axis1low)) * width;
            const y = top + ((input[1] - axis2low) / (axis2high - axis2low)) * height;

            const outputs = test(input);
            ctx.fillStyle = "black";
            drawCircle(ctx, {x, y}, Math.min(width, height) / 100);
            ctx.fillStyle = this.neuralNetworks[0].interpolateColorPublic(color1, color2, outputs[0]);
            drawCircle(ctx, {x, y}, Math.min(width, height) / 120);
        })

        ctx.restore();

    }
    public testErrorTrials(inputsList: number[][], outputsList: number[][], power?: number){
        if (inputsList.length !== outputsList.length){
            console.error("Wrong number of inputs or outputs, expected: " + inputsList.length + " | received: " + outputsList.length)
            return 0
        }
        const p = power || 2;
        this.neuralNetworks.forEach((nn: NeuralNetwork) => {
            for (let i = 0; i < inputsList.length; i++){
                nn.run(inputsList[i]);
                nn.testError(outputsList[i], true, p);
            }
            nn.error = (nn.error / inputsList.length) ** (1 / p);
            nn.meanError /= inputsList.length;
        });
        
    }
    public resetError(){
        this.neuralNetworks.forEach(nn => {
            nn.error = 0;
            nn.meanError = 0;
        });
    }
    public createTrials(inputsList: number[][], outputFunction: (inputs: number[]) => number[]){
        this.trialInputsList = inputsList;
        this.trialOutputsList = [];

        this.trialInputsList.forEach(i => {
            this.trialOutputsList.push(outputFunction(i));
        });
    }
    public killLowestError(numOfNeuralNetworks: number){
        this.sort();
        this.neuralNetworks = this.neuralNetworks.slice(0, numOfNeuralNetworks);
    }
    public reproduceSurvivors(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number){
        let numOfSurvivors = this.neuralNetworks.length;

        // Keep all survivors unchanged - they earned their survival!
        // Clone them and mutate only the clones to fill the population
        let index = 0;
        while (this.neuralNetworks.length < this.numOfNeuralNetworks){
            let nn = this.neuralNetworks[index].clone()
            nn.mutate(numOfWeights, weightStrength, numOfBiases, biasesStrength);
            this.neuralNetworks.push(nn);
            index++;
            if(index >= numOfSurvivors){index = 0;}
        }
    }
    public runGeneration(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number): number {
        this.generation++;
        this.resetError(); // Reset at START so previous error is visible for drawing
        this.testErrorTrials(this.trialInputsList, this.trialOutputsList, this.trialPower);
        this.sort();
        const bestError = this.neuralNetworks[0].error;
        this.killLowestError(Math.floor(this.numOfNeuralNetworks / 2));
        this.reproduceSurvivors(numOfWeights, weightStrength, numOfBiases, biasesStrength);
        return bestError;
    }
    public sort(){
        this.neuralNetworks.sort((a, b) => a.error - b.error);
    }
    
    private numToColorBlack(num: number): string {
        if (num >= 0) {
            return "rgb(0, " + (Math.tanh(num)) * 255 + ",0)"
        }
        if (num < 0) {
            return "rgb(" + (-Math.tanh(num)) * 255 + ", 0 ,0)"

        }
        return ""
    }
    private numToColorWhite(num: number): string {
        if (num >= 0) {
            return "rgb(" + (1 - Math.tanh(num)) * 255 + ", 255," + (1 - Math.tanh(num)) * 255 + ")"
        }
        if (num < 0) {
            return "rgb(255, " + (1 + Math.tanh(num)) * 255 + "," + (1 + Math.tanh(num)) * 255 + ")"
        }
        return ""
    }
}
///////////////////////////////////Neural Network////////////////////////////////////////////////////////
export class NeuralNetwork {
    error:number = 0;
    meanError:number = 0;

    numOfLayers: number;
    inputSize: number;
    hiddenLayerSizes: number[];
    outputSize: number;
    startRandomBias: boolean = true;
    startRandomWeights: boolean = true;
    clampBiases: boolean = false;
    clampWeights: boolean = false;

    changeWeightAlpha: boolean = false;
    linePower: number = 4;

    layers: Layer[] = [];
    activationFunction: ActivationFunction = 'tanh';
    outputActivationFunction: ActivationFunction = 'tanh';
    constructor(inputSize: number, hiddenLayerSizes: number[], outputSize: number, activationFunction: ActivationFunction, outputActivationFunction:ActivationFunction) {
        this.numOfLayers = hiddenLayerSizes.length + 2;
        this.inputSize = inputSize
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.initLayers();
        this.activationFunction = activationFunction;
        this.outputActivationFunction = outputActivationFunction;
    }
    public run(inputs: number[]) {
        if (inputs.length != this.inputSize) {
            throw new Error('Wrong Number of INPUTs, Expected: ' + this.inputSize + "| Received: " + inputs.length)
        }
        for (let i = 0; i < this.inputSize; i++){
            this.layers[0].neurons[i].value = inputs[i];
        }
        for (let l = 1; l < this.numOfLayers; l++){
            let neurons = this.layers[l].neurons;
            for (let n = 0; n < neurons.length; n++){
                let neuron = neurons[n];
                neuron.value = 0;
                let weights = neuron.weights;
                for (let w = 0; w < weights.length; w++){
                    let weight: Weight = weights[w];
                    neuron.value += weight.value * this.getNodeValue(weight.to);
                }
                neuron.value += neuron.bias;
                neuron.activate();
                
            }
        }
        return this.layers[this.layers.length-1];
    }
    public mutate(numOfWeights:number, weightStrength:number, numOfBiases:number, biasesStrength:number){
        for (let i = 0; i < numOfWeights; i++){
            let randLayerIndex = Math.floor(Math.random() * (this.numOfLayers - 1)) + 1;
            let neurons = this.layers[randLayerIndex].neurons
            let randNeuronIndex = Math.floor(Math.random() * neurons.length)
            let weights = this.layers[randLayerIndex].neurons[randNeuronIndex].weights
            let randWeightsIndex = Math.floor(Math.random() * weights.length);
            weights[randWeightsIndex].value += (Math.random() * 2 - 1) * weightStrength;
            if (this.clampWeights){
                weights[randWeightsIndex].value = Math.min(Math.max(weights[randWeightsIndex].value, -1), 1)
            }
        }
        for (let i = 0; i < numOfBiases; i++){
            let randLayerIndex = Math.floor(Math.random() * (this.numOfLayers - 1)) + 1;
            let neurons = this.layers[randLayerIndex].neurons
            let randNeuronIndex = Math.floor(Math.random() * neurons.length)
            neurons[randNeuronIndex].bias += (Math.random() * 2 - 1) * biasesStrength;
            if (this.clampBiases){
                neurons[randNeuronIndex].bias = Math.min(Math.max(neurons[randNeuronIndex].bias, -1), 1)
            }
        }
    }
    public testError(outputs: number[], additive: boolean, power?:number){
        let sum = 0;
        let meanSum = 0;
        const p = power || 2;
        for (let i = 0; i < this.outputSize; i++){
            const diff = Math.abs(this.getOutputLayer().neurons[i].value - outputs[i]);
            sum += diff ** p;
            meanSum += diff;
        }
        if (additive) {
            this.error += sum;
            this.meanError += meanSum;
        } else {
            this.error = sum;
            this.meanError = meanSum;
        }
    }
    public backPropogate(outputs: number[]){
        
    }
    public getNodeValue(np: NeuronPosition): number {
        return this.layers[np.layer].neurons[np.index].value;
    }
    public getInputLayer(): Layer {
        return this.layers[0]
    }
    public getOutputLayer(): Layer {
        return this.layers[this.numOfLayers - 1]
    }
    public clone(): NeuralNetwork {
        let nn = new NeuralNetwork(this.inputSize, this.hiddenLayerSizes, this.outputSize, this.activationFunction, this.outputActivationFunction);

        nn.startRandomBias = this.startRandomBias;
        nn.startRandomWeights = this.startRandomWeights;
        nn.clampBiases = this.clampBiases;
        nn.clampWeights = this.clampWeights;
        nn.changeWeightAlpha = this.changeWeightAlpha;
        nn.linePower = this.linePower;
        nn.error = this.error;
        nn.meanError = this.meanError;

        for (let l = 0; l < this.layers.length; l++) {
            const sourceLayer = this.layers[l];
            const targetLayer = nn.layers[l];

            for (let n = 0; n < sourceLayer.neurons.length; n++) {
                const sourceNeuron = sourceLayer.neurons[n];
                const targetNeuron = targetLayer.neurons[n];

                targetNeuron.value = sourceNeuron.value;
                targetNeuron.bias = sourceNeuron.bias;

                for (let w = 0; w < sourceNeuron.weights.length; w++) {
                    targetNeuron.weights[w].value = sourceNeuron.weights[w].value;
                }
            }
        }

        return nn;
    }
    private initLayers(): void {
        this.initNextLayer(this.inputSize, false, this.activationFunction);
        for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
            this.initNextLayer(this.hiddenLayerSizes[i], this.startRandomBias, this.activationFunction)
        }
        this.initNextLayer(this.outputSize, this.startRandomBias, this.outputActivationFunction);
    }
    private initNextLayer(numOfNeurons: number, randomBias: boolean, activationFunction:ActivationFunction) {
        const layerNeurons: Neuron[] = [];
        for (let j = 0; j < numOfNeurons; j++) {
            const neuronWeights: Weight[] = this.initWeights();

            layerNeurons.push(new Neuron(neuronWeights, (randomBias ? Math.random() * 2 - 1 : 0), activationFunction));
        }
        this.layers.push(new Layer(layerNeurons));
    }
    private initWeights(): Weight[] {
        if (this.layers.length == 0) { return []; }
        let w: Weight[] = [];
        let lastLayerIndex = this.layers.length - 1;
        for (let i = 0; i < this.layers[lastLayerIndex].neurons.length; i++) {
            w.push(new Weight((this.startRandomWeights ? Math.random() * 2 - 1 : 0), new NeuronPosition(lastLayerIndex, i)));
        }
        return w;
    }
    public write(): void {
        console.log(this.layers);
        for (let i = 0; i < this.layers.length; i++) {
            let str = ""
            for (let j = 0; j < this.layers[i].neurons.length; j++) {
                let neuron = this.layers[i].neurons[j];
                str += (neuron.bias)
                str += " "
            }
        }
    }
    public draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, displayErrorDigits?: number, displayMeanError?: boolean) {
        let spaceX = width / (this.numOfLayers + 1);
        let lastYs: number[] = [];

        const bottomPadding = (displayErrorDigits ? 15 : 0) + (displayMeanError ? 15 : 0);

        for (let i = 0; i < this.numOfLayers; i++) {
            let x: number = (i + 1) * spaceX + left;

            let lastX: number = i * spaceX + left;

            let neurons: Neuron[] = this.layers[i].neurons;
            let spaceY: number = (height - bottomPadding) / (neurons.length + 1);
            let newYs: number[] = []

            for (let j = 0; j < neurons.length; j++) {
                let neuron: Neuron = neurons[j];
                let y: number = (j + 1) * spaceY + top;
                newYs.push(y);

                for (let w = 0; w < lastYs.length; w++) {
                    ctx.save()
                    ctx.strokeStyle = this.numToColorBlack(neuron.weights[w].value);
                    ctx.lineWidth = 2;
                    if (this.changeWeightAlpha) ctx.globalAlpha = Math.abs(neuron.weights[w].value)**this.linePower;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(lastX, lastYs[w]);
                    ctx.stroke();

                    ctx.restore();
                }

            }
            lastYs = newYs;


        }
        for (let i = 0; i < this.numOfLayers; i++) {
            let neurons: Neuron[] = this.layers[i].neurons;
            let spaceY: number = (height - bottomPadding) / (neurons.length + 1);
            let x: number = (i + 1) * spaceX + left;

            for (let j = 0; j < neurons.length; j++) {
                let neuron: Neuron = neurons[j];
                let y: number = (j + 1) * spaceY + top;

                let r1 = Math.min(spaceX, spaceY) / 2.1
                ctx.save();
                ctx.fillStyle = this.numToColorBlack(neuron.bias);
                drawCircle(ctx, { x, y }, r1);
                ctx.restore();

                let r2 = Math.min(spaceX, spaceY) / 2.3
                ctx.save();
                ctx.fillStyle = this.numToColorWhite(neuron.value);
                drawCircle(ctx, { x, y }, r2);
                ctx.restore();

 

                ctx.save();
                let val = Math.floor(neuron.value * 1000) / 1000;
                let textStr = val.toString();
                let drawn: boolean = false;
                ctx.font = '12px sans-serif';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#000000';
                while (textStr.length > 0 && textStr != "-" && !drawn){
                    if (textStr.at(-1) == "."){
                        textStr = textStr.slice(0, -1);
                        continue;
                    }
                    let measure = ctx.measureText(textStr);
                    let w = measure.width;

                    if(w >= r2 * 1.8) {
                        textStr = textStr.slice(0, -1);
                        continue;
                    }

                    drawn = true;
                    ctx.fillText(textStr, x - w / 2, y);
                    ctx.restore();

                }

            }
        }
        if (displayErrorDigits){
            ctx.font = '12px sans-serif';
            ctx.fillStyle = '#000000';
            let textStr = "ϵ: " + (Math.floor(this.error * 10**displayErrorDigits) / 10**displayErrorDigits).toString()
            let drawn = false;
            while (textStr.length > 0 && textStr != "-" && !drawn){
                let measure = ctx.measureText(textStr);
                let w = measure.width;
                if(w >= width) {
                    textStr = textStr.slice(0, -1);
                    continue;
                }
                drawn = true;
                const yOffset = displayMeanError ? 20 : 10;
                ctx.fillText(textStr, left + width / 2 - w / 2,  top + height - yOffset);
                ctx.restore();
            }
        }
        if (displayMeanError){
            ctx.font = '12px sans-serif';
            ctx.fillStyle = '#000000';
            let textStr = "μ: " + (Math.floor(this.meanError * 10**displayErrorDigits) / 10**displayErrorDigits).toString()
            let drawn = false;
            while (textStr.length > 0 && textStr != "-" && !drawn){
                let measure = ctx.measureText(textStr);
                let w = measure.width;
                if(w >= width) {
                    textStr = textStr.slice(0, -1);
                    continue;
                }
                drawn = true;
                ctx.fillText(textStr, left + width / 2 - w / 2,  top + height - 10);
                ctx.restore();
            }
        }
    }
    
    private displayGrid2Input1Output(ctx, left: number, top: number, width: number, height: number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows: number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, getCellValue: (input1: number, input2: number) => number, getColorValue: (cellValue: number) => number){
        const headerOffset = showHeaders ? 1 : 0;
        let spaceX = width / (columns + headerOffset);
        let spaceY = height / (rows + headerOffset);

        ctx.save();
        ctx.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#e0e0e0';
        ctx.fillRect(left, top, width, height);

        if (showHeaders) {
            for (let i = 0; i < columns; i++){
                let x = left + (i + 1) * spaceX;
                let y = top;
                let headerValue = axis1low + i * (axis1high - axis1low) / (columns - 1);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            for (let j = 0; j < rows; j++){
                let x = left;
                let y = top + (j + 1) * spaceY;
                let headerValue = axis2low + j * (axis2high - axis2low) / (rows - 1);

                ctx.fillStyle = '#e0e0e0';
                ctx.fillRect(x, y, spaceX, spaceY);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            ctx.fillStyle = '#c0c0c0';
            ctx.fillRect(left, top, spaceX, spaceY);
        }

        for (let i = 0; i < columns; i++){
            for (let j = 0; j < rows; j++){
                let x = left + (i + headerOffset) * spaceX;
                let y = top + (j + headerOffset) * spaceY;
                let input1 = axis1low + i * (axis1high - axis1low) / (columns - 1);
                let input2 = axis2low + j * (axis2high - axis2low) / (rows - 1);

                let cellValue = getCellValue(input1, input2);
                ctx.fillStyle = this.numToColorWhite(getColorValue(cellValue));
                ctx.fillRect(x, y, spaceX + (showText? -1: 1), spaceY + (showText? -1: 1));

                if(showText){
                    ctx.fillStyle = '#000000';
                    let text = cellValue.toFixed(decimals);
                    let textWidth = ctx.measureText(text).width;
                    if (textWidth > spaceX * 0.9) {
                        text = cellValue.toFixed(Math.max(0, decimals - 1));
                        textWidth = ctx.measureText(text).width;
                        if (textWidth > spaceX * 0.9) {
                            text = cellValue.toFixed(0);
                        }
                    }
                    ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
                }
            }
        }
        ctx.restore();
        
    }
    
    public display2Input1OutputError (ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, errorRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, test: (inputs: number[]) => number[]){
        this.displayGrid2Input1Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders,
            (input1, input2) => {
                let val = this.run([input1, input2]);
                return val.neurons[0].value - test([input1, input2])[0];
            },
            (cellValue) => cellValue * errorRange
        );
    }

    public display2Input1OutputTest(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, ouputMiddle:number, outputRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, test: (inputs: number[]) => number[]){
        this.displayGrid2Input1Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders,
            (input1, input2) => test([input1, input2])[0],
            (cellValue) => cellValue * outputRange - ouputMiddle
        );
    }

    public display2Input1Output(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, ouputMiddle:number, outputRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean){
        this.displayGrid2Input1Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders,
            (input1, input2) => this.run([input1, input2]).neurons[0].value,
            (cellValue) => cellValue * outputRange - ouputMiddle
        );
    }

    private numToColorBlack(num: number): string {
        if (num >= 0) {
            return "rgb(0, " + (Math.tanh(num)) * 255 + ",0)"
        }
        if (num < 0) {
            return "rgb(" + (-Math.tanh(num)) * 255 + ", 0 ,0)"

        }
        return ""
    }
    private numToColorWhite(num: number): string {
        if (num >= 0) {
            return "rgb(" + (1 - Math.tanh(num)) * 255 + ", 255," + (1 - Math.tanh(num)) * 255 + ")"
        }
        if (num < 0) {
            return "rgb(255, " + (1 + Math.tanh(num)) * 255 + "," + (1 + Math.tanh(num)) * 255 + ")"
        }
        return ""
    }

    private interpolateColor(color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, weight: number): string {
        // weight should be between 0 and 1
        // weight = 0 means 100% color2, weight = 1 means 100% color1
        const clampedWeight = Math.max(0, Math.min(1, weight));
        const r = Math.round(color1.r * clampedWeight + color2.r * (1 - clampedWeight));
        const g = Math.round(color1.g * clampedWeight + color2.g * (1 - clampedWeight));
        const b = Math.round(color1.b * clampedWeight + color2.b * (1 - clampedWeight));
        return `rgb(${r}, ${g}, ${b})`;
    }

    public interpolateColorPublic(color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, weight: number): string {
        return this.interpolateColor(color1, color2, weight);
    }

    private displayGrid2Input2Output(ctx, left: number, top: number, width: number, height: number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows: number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, getOutputs: (input1: number, input2: number) => number[]){
        const headerOffset = showHeaders ? 1 : 0;
        let spaceX = width / (columns + headerOffset);
        let spaceY = height / (rows + headerOffset);

        ctx.save();
        ctx.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#e0e0e0';
        ctx.fillRect(left, top, width, height);

        if (showHeaders) {
            for (let i = 0; i < columns; i++){
                let x = left + (i + 1) * spaceX;
                let y = top;
                let headerValue = axis1low + i * (axis1high - axis1low) / (columns - 1);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            for (let j = 0; j < rows; j++){
                let x = left;
                let y = top + (j + 1) * spaceY;
                let headerValue = axis2low + j * (axis2high - axis2low) / (rows - 1);

                ctx.fillStyle = '#e0e0e0';
                ctx.fillRect(x, y, spaceX, spaceY);

                ctx.fillStyle = '#000000';
                let text = headerValue.toFixed(2);
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(1);
                }
                if (ctx.measureText(text).width > spaceX * 0.9) {
                    text = headerValue.toFixed(0);
                }
                ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
            }

            ctx.fillStyle = '#c0c0c0';
            ctx.fillRect(left, top, spaceX, spaceY);
        }

        for (let i = 0; i < columns; i++){
            for (let j = 0; j < rows; j++){
                let x = left + (i + headerOffset) * spaceX;
                let y = top + (j + headerOffset) * spaceY;
                let input1 = axis1low + i * (axis1high - axis1low) / (columns - 1);
                let input2 = axis2low + j * (axis2high - axis2low) / (rows - 1);

                let outputs = getOutputs(input1, input2);
                // outputs[0] is probability of class 1, outputs[1] is probability of class 2
                // Use outputs[0] as the weight for interpolation
                ctx.fillStyle = this.interpolateColor(color1, color2, outputs[0]);
                ctx.fillRect(x, y, spaceX + (showText? -1: 1), spaceY + (showText? -1: 1));

                if(showText){
                    ctx.fillStyle = '#000000';
                    let text = `${outputs[0].toFixed(decimals)}`;
                    let textWidth = ctx.measureText(text).width;
                    if (textWidth > spaceX * 0.9) {
                        text = outputs[0].toFixed(Math.max(0, decimals - 1));
                        textWidth = ctx.measureText(text).width;
                        if (textWidth > spaceX * 0.9) {
                            text = outputs[0].toFixed(0);
                        }
                    }
                    ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
                }
            }
        }
        ctx.restore();

    }

    public display2Input2Output(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}){
        this.displayGrid2Input2Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color1, color2,
            (input1, input2) => {
                let result = this.run([input1, input2]);
                return [result.neurons[0].value, result.neurons[1].value];
            }
        );
    }

    public display2Input2OutputTest(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, test: (inputs: number[]) => number[]){
        this.displayGrid2Input2Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, color1, color2,
            (input1, input2) => test([input1, input2])
        );
    }

    public display2Input2OutputError(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean, color1: {r: number, g: number, b: number}, color2: {r: number, g: number, b: number}, test: (inputs: number[]) => number[]){
        // For error visualization, use white (no error) to red (high error) gradient
        const errorColor1 = {r: 255, g: 255, b: 255}; // White for no error
        const errorColor2 = {r: 255, g: 0, b: 0}; // Red for high error

        this.displayGrid2Input2Output(ctx, left, top, width, height, axis1low, axis2low, axis1high, axis2high, rows, columns, decimals, showText, showHeaders, errorColor1, errorColor2,
            (input1, input2) => {
                let nnOutput = this.run([input1, input2]);
                let testOutput = test([input1, input2]);
                let error0 = Math.abs(nnOutput.neurons[0].value - testOutput[0]);
                let error1 = Math.abs(nnOutput.neurons[1].value - testOutput[1]);
                // Calculate total error (sum of both output errors)
                let totalError = (error0 + error1) / 2; // Average error
                // Return total error as both outputs for color interpolation
                // This will create a gradient from white (0 error) to red (1 error)
                return [totalError, 0]; // Using totalError as weight
            }
        );
    }

}
export type ActivationFunction = 'sigmoid' | 'relu' | 'tanh';

class Layer {
    neurons: Neuron[];
    constructor(neurons: Neuron[]) {
        this.neurons = neurons;
    }
}
class Neuron {
    value: number;
    weights: Weight[];
    bias: number;
    activationFunction: ActivationFunction;
    constructor(weights: Weight[], bias: number, activationFunction: ActivationFunction) {
        this.value = 0;
        this.weights = weights;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }
    public activate(){
        switch (this.activationFunction){
            case "relu":
                this.value = (this.value <= 0)? 0: this.value;
            case "sigmoid":
                this.value = 1 / (1 + Math.pow(Math.E, -this.value));
            case "tanh":
               this.value = Math.tanh(this.value);

        }
    }
}
class Weight {
    value: number;
    to: NeuronPosition;
    constructor(value: number, to: NeuronPosition) {
        this.value = value;
        this.to = to;
    }
}
class NeuronPosition {
    layer: number;
    index: number;
    constructor(layer: number, index: number) {
        this.layer = layer;
        this.index = index;
    }
}


