let video, canvas, ctx, stream;
let cap, src, dst, gray, rgb, tracker;
let arucoDict, arucoParams, arucoDetector;
let currentMode = "marker";
let isTracking = false;
let streaming = false;
let mouseX = 0,
  mouseY = 0;
let isMouseOver = false;
let syntheticMarker = null;

const statusText = document.getElementById("status");
const instructionText = document.getElementById("instruction-text");

// startup hook
window.startTracker = function () {
  startCamera();
};

async function startCamera() {
  video = document.getElementById("webcam");
  canvas = document.getElementById("canvas-output");
  ctx = canvas.getContext("2d");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });
    video.srcObject = stream;
    video.play();
  } catch (err) {
    statusText.innerText = "Camera Error: " + err;
    console.error(err);
    return;
  }

  video.addEventListener("canplay", () => {
    if (!streaming) {
      streaming = true;
      initOpenCV();
      requestAnimationFrame(processVideo);
      statusText.innerText = "System Running";
      instructionText.innerText = "Marker Mode: Show ArUco marker";
    }
  });
}

function initOpenCV() {
  let width = video.videoWidth;
  let height = video.videoHeight;
  canvas.width = width;
  canvas.height = height;

  cap = new cv.VideoCapture(video);
  src = new cv.Mat(height, width, cv.CV_8UC4);
  dst = new cv.Mat(height, width, cv.CV_8UC4);
  gray = new cv.Mat();
  rgb = new cv.Mat();
}

function processVideo() {
  if (!streaming) return;
  try {
    cap.read(src);
    src.copyTo(dst);

    if (currentMode === "marker") {
      if (!arucoDetector) {
        try {
          arucoParams = new cv.aruco_DetectorParameters();
          let arucoDict = cv.getPredefinedDictionary(
            cv.aruco_PredefinedDictionaryType.DICT_4X4_250.value,
          );
          arucoRefParams = new cv.aruco_RefineParameters(10.0, 3.0, true);
          arucoDetector = new cv.aruco_ArucoDetector(
            arucoDict,
            arucoParams,
            arucoRefParams,
          );
          const markerIdx = Math.floor(Math.random() * 250);
          syntheticMarker = new cv.Mat();
          arucoDict.generateImageMarker(markerIdx, 100, syntheticMarker);
        } catch (e) {
          console.warn("ArUco setup failed:", e);
        }
      }

      if (arucoDetector) {
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
        if (isMouseOver && syntheticMarker) {
          let mSize = 100;
          let x = Math.floor(mouseX - mSize / 2);
          let y = Math.floor(mouseY - mSize / 2);

          // Boundary checks to prevent crashing
          if (
            x >= 0 &&
            y >= 0 &&
            x + mSize < gray.cols &&
            y + mSize < gray.rows
          ) {
            let roi = gray.roi(new cv.Rect(x, y, mSize, mSize));
            syntheticMarker.copyTo(roi);
            roi.delete();

            let markerRGBA = new cv.Mat();
            cv.cvtColor(syntheticMarker, markerRGBA, cv.COLOR_GRAY2RGBA);
            let dstRoi = dst.roi(new cv.Rect(x, y, mSize, mSize));
            markerRGBA.copyTo(dstRoi);
            markerRGBA.delete();
            dstRoi.delete();
          } else {
          }
        }
        let corners = new cv.MatVector();
        let ids = new cv.Mat();
        let rejected = new cv.MatVector();

        // Detect
        arucoDetector.detectMarkers(gray, corners, ids, rejected);

        if (ids.rows > 0) {
          // MARKERS FOUND
          if (instructionText.innerText !== "Marker detected!") {
            instructionText.innerText = "Marker detected!";
          }

          for (let i = 0; i < ids.rows; ++i) {
            let markerObj = corners.get(i);
            let data = markerObj.data32F;

            for (let j = 0; j < 4; j++) {
              let p1 = new cv.Point(data[j * 2], data[j * 2 + 1]);
              let p2 = new cv.Point(
                data[((j + 1) % 4) * 2],
                data[((j + 1) % 4) * 2 + 1],
              );
              cv.line(dst, p1, p2, [0, 255, 0, 255], 2);
            }

            let firstCorner = new cv.Point(data[0], data[1]);
            cv.putText(
              dst,
              "Tracking ID#" + ids.data32S[i],
              new cv.Point(firstCorner.x, firstCorner.y - 10),
              cv.FONT_HERSHEY_SIMPLEX,
              0.5,
              [0, 255, 0, 255],
              2,
            );
          }
        } else {
          if (instructionText.innerText !== "no markers found") {
            instructionText.innerText = "no markers found";
          }
        }

        corners.delete();
        ids.delete();
        rejected.delete();
      }
    } else if (currentMode === "markerless" && isTracking && tracker) {
      cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
      let result = null;
      try {
        result = tracker.update(rgb);
      } catch (e) {
        console.error("Tracker update error:", e);
        isTracking = false;
      }

      if (result && result.length === 2 && result[0] === true) {
        let rect = result[1];
        let p1 = new cv.Point(rect.x, rect.y);
        let p2 = new cv.Point(rect.x + rect.width, rect.y + rect.height);

        cv.rectangle(dst, p1, p2, [0, 255, 0, 255], 2);
        cv.putText(
          dst,
          "Tracking",
          new cv.Point(rect.x, rect.y - 10),
          cv.FONT_HERSHEY_SIMPLEX,
          0.5,
          [0, 255, 0, 255],
          2,
        );
      } else {
        cv.putText(
          dst,
          "Lost",
          new cv.Point(20, 50),
          cv.FONT_HERSHEY_SIMPLEX,
          0.7,
          [255, 0, 0, 255],
          2,
        );
      }
    }
    cv.imshow("canvas-output", dst);
    requestAnimationFrame(processVideo);
  } catch (err) {
    console.error("Frame processing error:", err);
  }
}

window.setMode = function (mode) {
  currentMode = mode;
  isTracking = false;
  if (tracker) {
    try {
      tracker.delete();
    } catch (e) {}
    tracker = null;
  }

  document.getElementById("btn-marker").className =
    mode === "marker" ? "btn btn-active" : "btn";
  document.getElementById("btn-markerless").className =
    mode === "markerless" ? "btn btn-active" : "btn";

  instructionText.innerText =
    mode === "marker"
      ? "Marker Mode: Show ArUco marker"
      : "Markerless Mode: Click video to track object";
};

const canvasOutput = document.getElementById("canvas-output");
canvasOutput.addEventListener("mousedown", (e) => {
  if (currentMode !== "markerless") return;

  const rect = e.target.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (e.target.width / rect.width);
  const y = (e.clientY - rect.top) * (e.target.height / rect.height);

  if (tracker) {
    try {
      tracker.delete();
    } catch (e) {}
  }

  try {
    if (cv.TrackerMIL) {
      tracker = new cv.TrackerMIL();
    } else {
      alert("Error: TrackerMIL missing in this OpenCV build.");
      return;
    }

    let roi = new cv.Rect(x - 50, y - 50, 100, 100);
    tracker.init(src, roi);
    isTracking = true;
  } catch (err) {
    alert("Tracker Init Error: " + err);
    console.error(err);
  }
});

canvasOutput.addEventListener("mousemove", (e) => {
  console.log("isMouseOver");
  const rect = e.target.getBoundingClientRect();
  const scaleX = e.target.width / rect.width;
  const scaleY = e.target.height / rect.height;

  mouseX = (e.clientX - rect.left) * scaleX;
  mouseY = (e.clientY - rect.top) * scaleY;
  isMouseOver = true;
});

canvasOutput.addEventListener("mouseleave", () => {
  isMouseOver = false;
});
