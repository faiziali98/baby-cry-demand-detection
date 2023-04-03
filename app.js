const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const bodyParser = require('body-parser')
const predictor = require('./ml-module/predictor');
const indexRouter = require('./routes/index');
const {initializeModels, imageCroper} = require('./ml-module/crop-face');

initializeModels('./ml-module/face-models');
const app = express();
app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.urlencoded({ extended: true }))

// Configure multer to store uploaded files in the 'uploads' directory
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    
    const option = req.body.option;
    
    const uploadDir = path.join(__dirname, '/ml-module/data/uploads');
    const destDir = path.join(uploadDir, option);

    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }

    cb(null, destDir);
  },
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  }
});
const upload = multer({ storage });

app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'jade');

// Display the file upload form
app.use('/', indexRouter);

// Handle the file upload
app.post('/upload', upload.single('file'), async (req, res) => {
  imageCroper(req.file.path);

  res.send(`
    <p style="font-size: 24px; font-weight: bold; color: #333;">File uploaded!</p>
    <button style="background-color: #4CAF50; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 16px;" onclick="window.location.href='/'">Back to Main</button>
  `);
});

const check_upload = multer({ dest: '/ml-module/data/tests/' });

app.post('/check', check_upload.single('file'), async (req, res) => {
  const file = req.file;
  await imageCroper(file.path);
  const prediction = await predictor(file.path);

  res.send(`
    <p style="font-size: 24px; font-weight: bold; color: #333;">${prediction}</p>
    <button style="background-color: #4CAF50; color: white; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 16px;" onclick="window.location.href='/'">Back to Main</button>
  `);
})

module.exports = app;
