let video;
let handPoseNet;
let pose = [];
let predictions = [];
let nn;
let currentState = "waiting";
let targetLabel;

// number of hand features detected = 21

function keyPressed() {
  targetLabel = key;
  console.log(targetLabel);
  console.log("Collecting");
  currentState = "collecting";
}

function modelLoaded() {
  console.log("Handpose net loaded");
}

function setup() {
  createCanvas(640, 480);

  video = createCapture(VIDEO);
  video.size(width, height);

  // initialize hand pose net
  handPoseNet = ml5.handpose(video, modelLoaded);
  // handPoseNet.on("predict", gotHandPoses);
  handPoseNet.on("predict", (results) => {
    predictions = results;
    gotHandPoses;
  });

  video.hide();

  const options = {
    inputs: 21 * 3,
    outputs: 4,
    task: "classification",
    debug: true,
  };

  nn = ml5.neuralNetwork(options); // initialize dense neural network
}

function gotHandPoses(poses) {
  pose = poses;

  if (poses.length > 0) {
    pose = poses[0].landmarks; // an array of xyz co-ordinates of hand
  }
}

function draw() {
  image(video, 0, 0, width, height);
  drawKeypoints();
}

// A function to draw ellipses over the detected keypoints
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
