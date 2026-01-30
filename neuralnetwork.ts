export class NeuralNetwork { 
    inputSize:number;
    layerSizes: number[];
    outputSize:number;
    
    layers: Layer[] = [];
    numOfInputs = -1;
    activationFunction: ActivationFunction = 'sigmoid';
    constructor(inputSize:number, layerSizes: number[], outputSize:number, activationFunction: ActivationFunction) {
        this.inputSize = inputSize
        this.layerSizes = layerSizes;
        this.outputSize = outputSize;
        this.initializeLayers();
        this.activationFunction = activationFunction;
    }
    private initializeLayers(): void {
        this.createNextLayer(this.inputSize);
        for (let i = 0; i < this.layerSizes.length; i++) {
            this.createNextLayer(this.layerSizes[i])
        }
        this.createNextLayer(this.outputSize);
    }
    private createNextLayer (numOfNeurons: number){
        const layerNeurons: Neuron[] = [];

        for (let j = 0; j < numOfNeurons; j++) {
            const neuronWeights: Weight[] = [];
            layerNeurons.push(new Neuron(neuronWeights, Math.random() * 2 - 1));
        }
        this.layers.push(new Layer(layerNeurons));

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


