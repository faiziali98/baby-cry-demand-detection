const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

// Update the following variables with the path to your trained model and input image file
const modelPath = './ml-module/trained-models/model-1a/model.json';

const predictor = async (imagePath) => {

    // Load the trained model
    const model = await tf.loadLayersModel(`file://${modelPath}`);

    // Load the input image
    const image = fs.readFileSync(imagePath);
    const tensor = tf.node.decodeImage(image, 3);
    const resized = tf.image.resizeBilinear(tensor, [200, 200]).toFloat();
    const normalized = tf.div(resized, tf.scalar(255));
    const batched = normalized.expandDims(0);

    // Make a prediction on the input image
    const prediction = model.predict(batched);

    // Convert the prediction to an array
    const predictionArray = prediction.arraySync()[0];

    // Update the following array with your own classes
    const classes = ["Change", "Comfort", "Food"];

    // Find the index of the highest probability class
    const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));

    // Print the predicted class
    return `The newborn wants ${classes[maxIndex]}.`;
};

module.exports = predictor;