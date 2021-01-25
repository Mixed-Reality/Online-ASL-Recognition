let video;
let handPoseNet;
let pose = [];
let predictions = [];
let nn;
let currentState = "waiting";
let targetLabel = null;

let collectingData = false;

// number of hand features detected = 21

function isvalidKey(pressedKey) {
  if (pressedKey.toLowerCase() >= "a" && pressedKey.toLowerCase() <= "z")
    return true;

  return false;
}

// check for key presses
function keyPressed() {
  // targetLabel = key;
  pressedKey = key;
  targetLabel = isvalidKey(pressedKey) ? pressedKey : null;
  // save collected data if 's' is pressed
  if (targetLabel.toLowerCase() === "s") {
    nn.saveData();
  } else if (targetLabel) {
    console.log("Collecting");
    console.log(targetLabel);
    currentState = "collecting";

    // Stop collecting after 5 seconds
    setTimeout(() => {
      console.log("Stopped Collecting");
      currentState = "waiting";
      targetLabel = null;
    }, 10000);
  }
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
    // console.log(results);
    gotHandPoses(results);
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

function gotHandPoses(result) {
  if (result.length > 0) {
    pose = result[0].landmarks; // co-ordinates of the features

    if (currentState == "collecting") {
      let inputs = []; // inputs for neural network

      for (let i = 0; i < pose.length; i++) {
        // Storing feature co-ordinates in 1-D array
        const x = parseInt(pose[i][0].toFixed(2));
        const y = parseInt(pose[i][1].toFixed(2));
        const z = parseInt(pose[i][2].toFixed(2));
        inputs.push(x);
        inputs.push(y);
        inputs.push(z);
      }
      let target = [targetLabel];
      nn.addData(inputs, target); // Add the data to neural network raw array
    }
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
