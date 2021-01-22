let video;
let predictions = [];

// width = 640;
// height = 480;

function modelLoaded() {
  console.log("Model loaded");
}

function setup() {
  createCanvas(640, 480);
  // const video = document.getElementById("video");
  // video = document.querySelector("#videoElement");
  // if (navigator.mediaDevices.getUserMedia) {
  //   navigator.mediaDevices
  //     .getUserMedia({ video: true })
  //     .then(function (stream) {
  //       video.srcObject = stream;
  //     })
  //     .catch(function (err0r) {
  //       console.log("Something went wrong!");
  //     });
  // }

  video = createCapture(VIDEO);
  video.size(width, height);

  // Initialize model
  const handpose = ml5.handpose(video, modelLoaded);

  // This sets up an event that fills the global variable "predictions"
  // with an array every time new hand poses are detected
  handpose.on("predict", (results) => {
    predictions = results;
    console.log(predictions);
  });
  video.hide();
  console.log("Setup complete");
}

function draw() {
  image(video, 0, 0, width, height);

  drawKeypoints();
}

// Totally my code
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];
    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
  }
}
