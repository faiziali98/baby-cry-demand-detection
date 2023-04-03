const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const model = require('../model');
const { numClasses, imgHeight, imgWidth } = require('../constants');

// Update the following variables with the paths to your dataset
const dataDir = '../data/uploads';

// Update the following variables with the size of your images

const validationSplit = 0.2;

const imageList = fs.readdirSync(dataDir).map((dir, label) => {
  const dirPath = path.join(dataDir, dir);

  const dirImgs = fs.readdirSync(dirPath).map((filename) => {
    const imgPath = path.join(dirPath, filename);
    let img = tf.node.decodeImage(fs.readFileSync(imgPath), 3);

    if (img.shape[0] > imgWidth || img.shape[1] > imgHeight) {
      img = tf.image.resizeBilinear(img, [imgWidth, imgHeight]);
    }

    // Normalize the pixel values to be between -1 and 1
    img = img.div(255).sub(0.5).mul(2);
    return { xs: img, ys: label }
  });

  return dirImgs;
}).flat();


const numValidation = Math.floor(imageList.length * validationSplit);

const validationData = tf.data.generator(function*() {
    const validationImages = imageList.slice(0, numValidation);
    const xs = validationImages.map((image) => image.xs);
    const ys = validationImages.map((image) => image.ys);
    yield { xs: tf.stack(xs), ys: tf.oneHot(tf.tensor1d(ys, 'int32'), numClasses)};
});
  
const trainData = tf.data.generator(function*() {
    const trainingImages = imageList.slice(numValidation, imageList.length);
    const xs = trainingImages.map((image) => image.xs);
    const ys = trainingImages.map((image) => image.ys);
    yield { xs: tf.stack(xs), ys: tf.oneHot(tf.tensor1d(ys, 'int32'), numClasses)};
});
  
// Train the model
const numEpochs = 10;
const batchSize = 32;
const stepsPerEpoch = 100;
const validationSteps = 10;


model.fitDataset(trainData, {
  epochs: numEpochs,
  batchSize: batchSize,
  stepsPerEpoch: stepsPerEpoch,
  validationData: validationData,
  validationSteps: validationSteps
}).then(() => {
  // Save the trained model to a file
  model.save('file://./trained-models/model-1c');
});
