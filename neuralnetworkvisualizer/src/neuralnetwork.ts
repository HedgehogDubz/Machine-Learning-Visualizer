import { drawCircle } from './graphics';
import type { Point } from './graphics';
export class NeuralNetwork {
    numOfLayers: number;
    inputSize: number;
    hiddenLayerSizes: number[];
    outputSize: number;
    startRandomBias: boolean = true;
    startRandomWeights: boolean = true;
    linePower = 4;

    layers: Layer[] = [];
    activationFunction: ActivationFunction = 'sigmoid';
    constructor(inputSize: number, hiddenLayerSizes: number[], outputSize: number, activationFunction: ActivationFunction) {
        this.numOfLayers = hiddenLayerSizes.length + 2;
        this.inputSize = inputSize
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.initLayers();
        this.activationFunction = activationFunction;
    }
    private initLayers(): void {
        this.initNextLayer(this.inputSize, false);
        for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
            this.initNextLayer(this.hiddenLayerSizes[i], true)
        }
        this.initNextLayer(this.outputSize, false);
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
            console.log(str);
        }
    }
    public draw(ctx: CanvasRenderingContext2D, left: number, top: number, width: number, height: number) {
        let spaceX = width / (this.numOfLayers + 1);
        let lastYs: number[] = [];

        let lastSpaceY = 0;
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
                    ctx.globalAlpha = Math.abs(neuron.weights[w].value)**this.linePower;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.lineTo(lastX, lastYs[w]);
                    ctx.stroke();

                    ctx.restore();
                }

            }
            lastYs = newYs;
            lastSpaceY = spaceY;


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
                ctx.font = '12px sans-serif';
                let measure = ctx.measureText(textStr);
                let w = measure.width;
                let h = measure.emHeightAscent;
                ctx.fillText(textStr, x - w / 2,  y + h / 2);
                ctx.restore();

            }
        }
    }
    numToColorBlack(num: number): string {
        if (num >= 0) {
            return "rgb(0, " + (Math.tanh(num)) * 255 + ",0)"
        }
        if (num < 0) {
            return "rgb(" + (-Math.tanh(num)) * 255 + ", 0 ,0)"

        }
        return ""
    }
    numToColorWhite(num: number): string {
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


