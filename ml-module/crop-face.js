const faceapi = require('face-api.js');
const fs = require('fs');
const canvas = require('canvas');

// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })

const imageCroper = async (imgPath) => {
  const inputImg = await canvas.loadImage(imgPath);

  // Detect face in image
  const faceDetectionOptions = new faceapi.TinyFaceDetectorOptions({ 
    inputSize: 320, 
    scoreThreshold: 0.5 
  });
  const faceDetectionResult = await faceapi.detectSingleFace(inputImg, faceDetectionOptions).withFaceLandmarks();

  if (!faceDetectionResult) {
    console.log('No face detected in input image');
    return;
  }

  const faceBoundingBox = faceDetectionResult.detection.box;

  // Crop image around face
  const croppedCanvas = canvas.createCanvas(200, 200);
  const croppedCtx = croppedCanvas.getContext('2d');
  
  croppedCtx.drawImage(inputImg, 
    Math.max(0, faceBoundingBox.x - faceBoundingBox.width * 0.5),
    Math.max(0, faceBoundingBox.y - faceBoundingBox.height * 0.5),
    faceBoundingBox.width * 2,
    faceBoundingBox.height * 2,
    0,
    0,
    200,
    200
  );

  // Save cropped image
  fs.writeFileSync(imgPath, croppedCanvas.toBuffer());

  console.log('Image cropped and saved to output.jpg');
}

const initializeModels = (modelsDir) => Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromDisk(modelsDir),
    faceapi.nets.faceLandmark68Net.loadFromDisk(modelsDir)
  ]);

module.exports = {initializeModels, imageCroper};