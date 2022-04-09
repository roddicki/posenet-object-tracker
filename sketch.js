// Grab elements, create settings, etc.
var video = document.getElementById("video");
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");

// canvas
var canvas0 = document.getElementById("canvas-one");
var ctx0 = canvas0.getContext("2d");
//canvas
var canvas1 = document.getElementById("canvas-two");
var ctx1 = canvas1.getContext("2d");

// The detected positions will be inside an array
let poses = [];
let person = [];

video.addEventListener('loadeddata', (event) => {
  console.log('vid loaded');
});


// Create a webcam capture
//if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
//  navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
//    video.srcObject = stream;
//    video.play();
//  });
//}


// A function to draw the video and poses into the canvas.
// This function is independent of the result of posenet
// This way the video will not seem slow if poseNet
// is not detecting a position
function drawCameraIntoCanvas() {
  // Draw the video element into the canvas
  ctx.drawImage(video, 0, 0, 640, 480);
  // We can call both functions to draw all keypoints and the skeletons
  //drawKeypoints();
  //drawSkeleton();
  detectPerson();
  window.requestAnimationFrame(drawCameraIntoCanvas);
}
// Loop over the drawCameraIntoCanvas function
//drawCameraIntoCanvas(); //uncomment for webcam

// image classifier
// Initialize the Image Classifier method with MobileNet
const objectDetector = ml5.objectDetector('cocossd', {}, objectModelLoaded);

// When the model is loaded
function objectModelLoaded() {
  console.log('cocossd Model Loaded');
  video.play();
  drawCameraIntoCanvas(); // comment out for webcam
}

// detect a person and track them
function detectPerson() {
  // Make an object prediction with a selected frame
  objectDetector.detect(video, (err, results) => {
    // draw rect over canvas
    ctx.beginPath();
    ctx.rect(0, 0, 640, 480);
    ctx.fillStyle = "rgba(255,255,255,0.7)";
    ctx.fill();
    ctx.lineWidth = "1";
    ctx.strokeStyle = "black";
    ctx.stroke();
    // if any of the predictions are a person
    for (var i = 0; i < results.length; i++) {
      if (results[i].label == "person") {
        // create person  if doesn't exist
        createPerson(i);
        // draw bounding box
        //context.drawImage(img,clipx,clipy,clipwidth,clipheight,x,y,width,height);
        //ctx.drawImage(video, results[i].x, results[i].y, results[i].width, results[i].height, results[i].x, results[i].y, results[i].width, results[i].height);
        if (i == 0) {
          ctx0.beginPath();
          ctx0.rect(0, 0, 640, 480);
          ctx0.fillStyle = "rgba(255,255,255,1)";
          ctx0.fill();
          ctx0.drawImage(video, results[i].x, results[i].y, results[i].width, results[i].height, results[i].x, results[i].y, results[i].width, results[i].height);
          drawKeypoints(ctx0);
          drawSkeleton(ctx0);
        // call classify here
        } 
        else if(i==1) {
          ctx1.beginPath();
          ctx1.rect(0, 0, 640, 480);
          ctx1.fillStyle = "rgba(255,255,255,1)";
          ctx1.fill();
          ctx1.drawImage(video, results[i].x, results[i].y, results[i].width, results[i].height, results[i].x, results[i].y, results[i].width, results[i].height);
          drawKeypoints(ctx1);
          drawSkeleton(ctx1);
        }
        ctx.font = "20px Arial";
        ctx.fillStyle = "green";
        ctx.fillText("ID:" + i, results[i].x, results[i].y+20);
        ctx.beginPath();
        ctx.lineWidth = "2";
        ctx.strokeStyle = "green";
        ctx.rect(results[i].x, results[i].y, results[i].width, results[i].height);
        ctx.stroke();
      }  
      
    }
  });
}

// if id doesn't already exists in person[] create a person
function createPerson(i) {
  const checkID = obj => obj.id === i;
  let IDexists = person.some(checkID);
  if (IDexists == false) {
    person.push({id:i, poses:[]});
  }
  console.log(JSON.stringify(person));
}

// Create a new poseNet method with a single detection
const poseNet = ml5.poseNet(video, modelReady);
poseNet.on("pose", gotPoses);

// A function that gets called every time there's an update from the model
function gotPoses(results) {
  poses = results;
}

// model ready
function modelReady() {
  console.log("pose model ready");
  poseNet.multiPose(video); // single pose check on each person
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i += 1) {
    // For each pose detected, loop through all the keypoints
    for (let j = 0; j < poses[i].pose.keypoints.length; j += 1) {
      let keypoint = poses[i].pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        ctx.beginPath();
        ctx.fillStyle = "#FF0000";
        ctx.arc(keypoint.position.x, keypoint.position.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i += 1) {
    // For every skeleton, loop through all body connections
    for (let j = 0; j < poses[i].skeleton.length; j += 1) {
      let partA = poses[i].skeleton[j][0];
      let partB = poses[i].skeleton[j][1];
      ctx.beginPath();
      ctx.moveTo(partA.position.x, partA.position.y);
      ctx.lineTo(partB.position.x, partB.position.y);
      ctx.strokeStyle = "#FF0000";
      ctx.stroke();
    }
  }
}
