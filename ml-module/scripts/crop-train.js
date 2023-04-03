const faceapi = require('face-api.js');
const { imageCroper, initializeModels } = require('../crop-face');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node'); //need fix here https://github.com/justadudewhohacks/face-api.js/issues/633

// Load face detection model
initializeModels('../face-models').then(detectFace);

async function detectFace() {
    // Load input image

    const dataDir = '../data/uploads';

    fs.readdirSync(dataDir).map((dir) => {
        const dirPath = path.join(dataDir, dir);

        fs.readdirSync(dirPath).map(async (filename) => {
        imageCroper(path.join(dirPath, filename));
        });
    });
}