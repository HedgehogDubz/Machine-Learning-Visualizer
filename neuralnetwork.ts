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

    constructor(numOfNeuralNetworks:number, inputSize: number, hiddenLayerSizes: number[], outputSize: number, activationFunction: ActivationFunction){
        this.numOfNeuralNetworks = numOfNeuralNetworks;
        for(let i = 0; i < numOfNeuralNetworks; i++){
            const nn = new NeuralNetwork(inputSize, hiddenLayerSizes, outputSize, activationFunction);
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
    public draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number, rowSize: number, displayErrorDigits?: number, displayMeanError?: boolean){
        let xSpace = width / rowSize;
        let ySpace = height / Math.ceil(this.neuralNetworks.length / rowSize);
        let col = 0;
        let row = 0;
        this.neuralNetworks.forEach((nn: NeuralNetwork) => {
            nn.draw(ctx, left + xSpace * col, top + ySpace * row, xSpace, ySpace, displayErrorDigits, displayMeanError);
            col++;
            if(col >= rowSize){
                col = 0;
                row++;
            }
        });
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
    constructor(inputSize: number, hiddenLayerSizes: number[], outputSize: number, activationFunction: ActivationFunction) {
        this.numOfLayers = hiddenLayerSizes.length + 2;
        this.inputSize = inputSize
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.initLayers();
        this.activationFunction = activationFunction;
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
                neuron.value = this.activate(neuron.value);
                
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
    public getNodeValue(np: NeuronPosition): number {
        return this.layers[np.layer].neurons[np.index].value;
    }
    public getInputLayer(): Layer {
        return this.layers[0]
    }
    public getOutputLayer(): Layer {
        return this.layers[this.numOfLayers - 1]
    }
    public activate(n: number){
        switch (this.activationFunction){
            case "relu":
                return (n <= 0)? 0: n;
            case "sigmoid":
                return 1 / (1 + Math.pow(Math.E, -n));
            case "tanh":
                return Math.tanh(n);

        }
    }
    public clone(): NeuralNetwork {
        let nn = new NeuralNetwork(this.inputSize, this.hiddenLayerSizes, this.outputSize, this.activationFunction);

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
        this.initNextLayer(this.inputSize, false);
        for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
            this.initNextLayer(this.hiddenLayerSizes[i], this.startRandomBias)
        }
        this.initNextLayer(this.outputSize, this.startRandomBias);
    }
    private initNextLayer(numOfNeurons: number, randomBias: boolean) {
        const layerNeurons: Neuron[] = [];
        for (let j = 0; j < numOfNeurons; j++) {
            const neuronWeights: Weight[] = this.initWeights();

            layerNeurons.push(new Neuron(neuronWeights, (randomBias ? Math.random() * 2 - 1 : 0)));
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
                    ctx.lineWidth = 5;
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
                while (textStr.length > 0 && textStr != "-" && !drawn){
                    if (textStr.at(-1) == "."){
                        textStr = textStr.slice(0, -1);
                        continue;
                    }
                    ctx.font = '12px sans-serif';
                    let measure = ctx.measureText(textStr);
                    let w = measure.width;
                    let h = measure.emHeightAscent;

                    if(w >= r2 * 2 || h >= r2 * 2) {
                        textStr = textStr.slice(0, -1);
                        continue;
                    }
                    
                    drawn = true;
                    ctx.fillText(textStr, x - w / 2,  y + h / 2);
                    ctx.restore();

                }

            }
        }
        if (displayErrorDigits){
            ctx.font = '12px sans-serif';
            let textStr = "ϵ: " + (Math.floor(this.error * 10**displayErrorDigits) / 10**displayErrorDigits).toString()
            let drawn = false;
            while (textStr.length > 0 && textStr != "-" && !drawn){
                let measure = ctx.measureText(textStr);
                let w = measure.width;
                let h = measure.emHeightAscent;
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
            let textStr = "μ: " + (Math.floor(this.meanError * 10**displayErrorDigits) / 10**displayErrorDigits).toString()
            let drawn = false;
            while (textStr.length > 0 && textStr != "-" && !drawn){
                let measure = ctx.measureText(textStr);
                let w = measure.width;
                let h = measure.emHeightAscent;
                if(w >= width) {
                    textStr = textStr.slice(0, -1);
                    continue;
                }
                drawn = true;
                ctx.fillText(textStr, left + width / 2 - w / 2,  top + height - h);
                ctx.restore();
            }
        }
    }
    public display2Input1Output(ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, ouputMiddle:number, outputRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean){
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
                let val = this.run([input1, input2]);

                ctx.fillStyle = this.numToColorWhite(val.neurons[0].value * outputRange - ouputMiddle);
                ctx.fillRect(x, y, spaceX + (showText? -1: 1), spaceY + (showText? -1: 1));

                if(showText){
                    ctx.fillStyle = '#000000';
                    let text = val.neurons[0].value.toFixed(decimals);
                    if (ctx.measureText(text).width > spaceX * 0.9) {
                        text = val.neurons[0].value.toFixed(Math.max(0, decimals - 1));
                    }
                    if (ctx.measureText(text).width > spaceX * 0.9) {
                        text = val.neurons[0].value.toFixed(0);
                    }
                    ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
                }
            }
        }

        ctx.restore();
    }
    public display2Input1OutputError (test: (inputs: number[]) => number[],ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, errorRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean){
        const headerOffset = showHeaders ? 1 : 0;
        let spaceX = width / (columns + headerOffset);
        let spaceY = height / (rows + headerOffset);

        ctx.save();
        ctx.font = `${Math.min(spaceX, spaceY) * 0.3}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        if (showHeaders) {
            for (let i = 0; i < columns; i++){
                let x = left + (i + 1) * spaceX;
                let y = top;
                let headerValue = axis1low + i * (axis1high - axis1low) / (columns - 1);

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
                let val = this.run([input1, input2]);

                let error = test([input1, input2])[0] - val.neurons[0].value;
                ctx.fillStyle = this.numToColorWhite(error * errorRange);
                ctx.fillRect(x, y, spaceX + (showText? -1: 1), spaceY + (showText? -1: 1));

                if(showText){
                    ctx.fillStyle = '#000000';
                    let text = error.toFixed(decimals);
                    if (ctx.measureText(text).width > spaceX * 0.9) {
                        text = error.toFixed(Math.max(0, decimals - 1));
                    }
                    if (ctx.measureText(text).width > spaceX * 0.9) {
                        text = error.toFixed(0);
                    }
                    ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
                }
            }
        }

        ctx.restore();
    }
    public display2Input1OutputTest(test: (inputs: number[]) => number[], ctx, left: number, top: number, width:number, height:number, axis1low: number, axis2low: number, axis1high: number, axis2high: number, ouputMiddle:number, outputRange: number, rows:number, columns: number, decimals: number, showText: boolean, showHeaders: boolean){
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
                ctx.fillStyle = this.numToColorWhite(test([input1, input2])[0] * outputRange - ouputMiddle);
                ctx.fillRect(x, y, spaceX + (showText? -1: 1), spaceY + (showText? -1: 1));

                if(showText){
                    ctx.fillStyle = '#000000';
                    let text = test([input1, input2])[0].toFixed(decimals);
                    if (ctx.measureText(text).width > spaceX * 0.9) {
                        text = test([input1, input2])[0].toFixed(Math.max(0, decimals - 1));
                    }
                    if (ctx.measureText(text).width > spaceX * 0.9) {
                        text = test([input1, input2])[0].toFixed(0);
                    }
                    ctx.fillText(text, x + spaceX / 2, y + spaceY / 2);
                }
            }
        }
        ctx.restore();
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
    constructor(weights: Weight[], bias: number) {
        this.value = 0;
        this.weights = weights;
        this.bias = bias;
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


