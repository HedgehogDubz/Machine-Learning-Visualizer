import { drawCircle } from './graphics.js';
import type { Point } from './graphics.js';
export class NeuralNetwork {
    numOfLayers: number;
    inputSize: number;
    hiddenLayerSizes: number[];
    outputSize: number;
    startRandomBias: boolean = false;
    startRandomWeights: boolean = false;
    clampBiases: boolean = true;
    clampWeights: boolean = true;

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
            let randLayerIndex = Math.floor(Math.random() * this.numOfLayers);
            let neurons = this.layers[randLayerIndex].neurons
            let randNeuronIndex = Math.floor(Math.random() * neurons.length)
            neurons[randNeuronIndex].bias += (Math.random() * 2 - 1) * biasesStrength;
            if (this.clampBiases){
                neurons[randNeuronIndex].bias = Math.min(Math.max(neurons[randNeuronIndex].bias, -1), 1)
            }
        }
    }
    public getNodeValue(np: NeuronPosition): number {
        return this.layers[np.layer].neurons[np.index].value;
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
    private initLayers(): void {
        this.initNextLayer(this.inputSize, false);
        for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
            this.initNextLayer(this.hiddenLayerSizes[i], this.startRandomBias)
        }
        this.initNextLayer(this.outputSize, this.startRandomBias);
    }
    private initNextLayer(numOfNeurons: number, randomBias: boolean) {
        const layerNeurons: Neuron[] = [];
        console.log(randomBias);
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
    public draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number) {
        let spaceX = width / (this.numOfLayers + 1);
        let lastYs: number[] = [];

        //weights
        for (let i = 0; i < this.numOfLayers; i++) {
            let x: number = (i + 1) * spaceX + left;

            let lastX: number = i * spaceX + left;

            let neurons: Neuron[] = this.layers[i].neurons;
            let spaceY: number = height / (neurons.length + 1);
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
        //neurons
        for (let i = 0; i < this.numOfLayers; i++) {
            let neurons: Neuron[] = this.layers[i].neurons;
            let spaceY: number = height / (neurons.length + 1);
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
type ActivationFunction = 'sigmoid' | 'relu' | 'tanh';

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


