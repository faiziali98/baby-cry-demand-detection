# Baby Demand Detector
## Overview
This is a Node.js application that collects data, trains a machine learning model, and uses the model to detect baby demand. The model is based on a deep neural network trained using TensorFlow.js.

## Directory Tree

```
baby-demand-detector/
├── data/
│   ├── feeding/
│   │   ├── feeding_1.jpg
│   │   ├── feeding_2.jpg
│   │   └── ...
│   ├── napping/
│   │   ├── napping_1.jpg
│   │   ├── napping_2.jpg
│   │   └── ...
│   ├── playing/
│   │   ├── playing_1.jpg
│   │   ├── playing_2.jpg
│   │   └── ...
│   └── ...
├── models/
│   ├── model.json
│   └── weights.bin
├── src/
│   ├── gatherData.js
│   ├── trainModel.js
│   └── detectDemand.js
├── node_modules/
├── package.json
├── package-lock.json
├── README.md
└── .gitignore
```

## How to Use
1. Clone the repository: git clone https://github.com/your-username/baby-demand-detector.git
2. Navigate to the project directory: cd baby-demand-detector
3. Install the required packages: npm install
4. Collect data by running the gatherData.js script: node src/gatherData.js
5. Train the model by running the trainModel.js script: node src/trainModel.js
6. Use the model to detect demand by running the detectDemand.js script: node src/detectDemand.js

*Note: Make sure to place your data in the data directory, with each class of data in a separate subdirectory (e.g. feeding, napping, playing, etc.). Also, ensure that your data is properly labeled with meaningful filenames.
