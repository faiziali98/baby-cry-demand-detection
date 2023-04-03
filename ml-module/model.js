const tf = require('@tensorflow/tfjs-node');
const { inputShape, numClasses } = require('./constants');

// Define the CNN architecture
const model = tf.sequential();
model.add(tf.layers.conv2d({
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
  inputShape: inputShape
}));
model.add(tf.layers.maxPooling2d({
  poolSize: 2,
  strides: 2
}));
model.add(tf.layers.conv2d({
  filters: 64,
  kernelSize: 3,
  activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: 2,
  strides: 2
}));
model.add(tf.layers.conv2d({
  filters: 128,
  kernelSize: 3,
  activation: 'relu'
}));
model.add(tf.layers.maxPooling2d({
  poolSize: 2,
  strides: 2
}));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({
  units: 128,
  activation: 'relu'
}));
model.add(tf.layers.dense({
  units: numClasses,
  activation: 'softmax'
}));
model.compile({
  loss: 'categoricalCrossentropy',
  optimizer: tf.train.adam(),
  metrics: ['accuracy']
});

module.exports = model;