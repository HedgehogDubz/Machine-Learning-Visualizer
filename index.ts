import { NeuralNetwork } from './neuralnetwork.js';

// Get canvas and context
const canvas = document.getElementById('canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d');

if (!ctx) {
    throw new Error('Could not get canvas context');
}

// Create a neural network with 3 layers: 3 input neurons, 4 hidden neurons, 2 output neurons
const neuralNetwork = new NeuralNetwork([3, 4, 2]);

// Function to draw the neural network
function drawNeuralNetwork(): void {
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const layers = neuralNetwork.getLayers();
    const layerSpacing = canvas.width / (layers.length + 1);
    const neuronRadius = 20;

    // Store neuron positions for drawing connections
    const neuronPositions: { x: number; y: number }[][] = [];

    // Draw connections first (so they appear behind neurons)
    const weights = neuralNetwork.getWeights();
    for (let i = 0; i < layers.length - 1; i++) {
        const currentLayerPositions: { x: number; y: number }[] = [];
        const nextLayerPositions: { x: number; y: number }[] = [];

        const currentLayerSize = layers[i];
        const nextLayerSize = layers[i + 1];
        const currentVerticalSpacing = canvas.height / (currentLayerSize + 1);
        const nextVerticalSpacing = canvas.height / (nextLayerSize + 1);

        // Calculate positions
        for (let j = 0; j < currentLayerSize; j++) {
            currentLayerPositions.push({
                x: layerSpacing * (i + 1),
                y: currentVerticalSpacing * (j + 1)
            });
        }

        for (let j = 0; j < nextLayerSize; j++) {
            nextLayerPositions.push({
                x: layerSpacing * (i + 2),
                y: nextVerticalSpacing * (j + 1)
            });
        }

        // Draw connections
        for (let j = 0; j < nextLayerSize; j++) {
            for (let k = 0; k < currentLayerSize; k++) {
                const weight = weights[i][j][k];
                const opacity = Math.abs(weight);
                const color = weight > 0 ? `rgba(76, 175, 80, ${opacity})` : `rgba(244, 67, 54, ${opacity})`;

                ctx.beginPath();
                ctx.moveTo(currentLayerPositions[k].x, currentLayerPositions[k].y);
                ctx.lineTo(nextLayerPositions[j].x, nextLayerPositions[j].y);
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }

        if (i === 0) neuronPositions.push(currentLayerPositions);
        neuronPositions.push(nextLayerPositions);
    }

    // Draw neurons
    for (let i = 0; i < layers.length; i++) {
        const layerSize = layers[i];
        const verticalSpacing = canvas.height / (layerSize + 1);

        for (let j = 0; j < layerSize; j++) {
            const x = layerSpacing * (i + 1);
            const y = verticalSpacing * (j + 1);

            // Draw neuron circle
            ctx.beginPath();
            ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
            ctx.fillStyle = '#667eea';
            ctx.fill();
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    }
}

// Draw the neural network
drawNeuralNetwork();

// Test the neural network with sample input
const testInput = [0.5, 0.3, 0.8];
const output = neuralNetwork.feedForward(testInput);
console.log('Neural Network Output:', output);

