import { drawCircle } from './graphics';
import type {Point} from './graphics';
export class NeuralNetwork { 
    numOfLayers:number;
    inputSize:number;
    hiddenLayerSizes: number[];
    outputSize:number;

    layers: Layer[] = [];
    activationFunction: ActivationFunction = 'sigmoid';
    constructor(inputSize:number, hiddenLayerSizes: number[], outputSize:number, activationFunction: ActivationFunction) {
        this.numOfLayers = hiddenLayerSizes.length + 2;
        this.inputSize = inputSize
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;
        this.initLayers();
        this.activationFunction = activationFunction;
    }
    private initLayers(): void {
        this.initNextLayer(this.inputSize);
        for (let i = 0; i < this.hiddenLayerSizes.length; i++) {
            this.initNextLayer(this.hiddenLayerSizes[i])
        }
        this.initNextLayer(this.outputSize);
    }
    private initNextLayer (numOfNeurons: number){
        const layerNeurons: Neuron[] = [];

        for (let j = 0; j < numOfNeurons; j++) {
            const neuronWeights: Weight[] = this.initWeights();

            layerNeurons.push(new Neuron(neuronWeights, 0));
        }
        this.layers.push(new Layer(layerNeurons));
    }
    private initWeights(): Weight[] {
        if (this.layers.length == 0) {return [];}
        let w:Weight[] = [];
        let lastLayerIndex = this.layers.length - 1;
        for (let i = 0; i < this.layers[lastLayerIndex].neurons.length; i++){
            w.push(new Weight(0, new NeuronPosition(lastLayerIndex, i)));
        }
        return w;
    }
    public write(): void {
        console.log(this.layers);
        for(let i = 0; i < this.layers.length; i++){
            let str = ""
            for(let j = 0; j < this.layers[i].neurons.length; j++){
                let neuron = this.layers[i].neurons[j];
                str += (neuron.bias)
                str += " "
            }
            console.log(str);
        }
    }
    public draw(ctx: CanvasRenderingContext2D, left:number, top:number, width:number, height:number){
        let spaceX = width / (this.numOfLayers + 1);
        let lastYs:number[] = [];

        for (let i = 0; i < this.numOfLayers; i++){
            let x:number = (i+1) * spaceX;

            let lastX:number = i * spaceX;

            let neurons: Neuron[] = this.layers[i].neurons;
            let spaceY: number = height / (neurons.length + 1);

            for (let j = 0; j < neurons.length; j++){
                let neuron: Neuron = neurons[j];
                let y:number = (j+1) * spaceY;
                drawCircle(ctx, {x, y}, Math.min(spaceX, spaceY));
            }



        }
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
    weights: Weight[];
    bias: number;
    constructor(weights: Weight[], bias: number) {
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


