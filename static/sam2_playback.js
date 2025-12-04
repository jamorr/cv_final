let sam2State = {
  video: null,
  canvas: null,
  ctx: null,
  data: null,
  fps: 30,
  isPlaying: false,
  reqId: null,
  mats: {
    src: null,
    dst: null,
    overlay: null,
    contoursVec: null,
  },
};

async function initSam2() {
  const container = document.getElementById("sam2-container");
  const status = document.getElementById("sam2-status");
  const btn = document.getElementById("btn-load-sam2");

  // UI Updates
  container.style.display = "block";
  btn.disabled = true;
  btn.innerText = "Loading Data...";
  status.innerText = "Fetching segmentation data...";

  sam2State.video = document.getElementById("sam2-video");
  sam2State.canvas = document.getElementById("sam2-canvas");
  sam2State.ctx = sam2State.canvas.getContext("2d");

  // Fetch Data
  try {
    const response = await fetch("/tracker/sam2_data");
    if (!response.ok) throw new Error("Network response was not ok");
    sam2State.data = await response.json();

    if (sam2State.data.fps) sam2State.fps = sam2State.data.fps;

    status.innerText = "Data Loaded. Ready to Play.";
    btn.style.display = "none"; // Hide load button

    setupSam2OpenCV();

    // Auto-play
    sam2State.video.play();
    requestAnimationFrame(renderSam2Frame);
  } catch (error) {
    console.error("SAM2 Init Error:", error);
    status.innerText = "Error loading data: " + error.message;
    btn.disabled = false;
    btn.innerText = "Retry SAM2 Demo";
  }
}

function setupSam2OpenCV() {
  // Initialize OpenCV Mats for the SAM2 pipeline
  // We assume video size is 640x480 based on the template,
  // but ideally we should wait for metadata.
  // For this demo we'll init on first frame render or assume fixed size.
  let width = 640;
  let height = 480;

  sam2State.canvas.width = width;
  sam2State.canvas.height = height;

  try {
    sam2State.mats.src = new cv.Mat(height, width, cv.CV_8UC4);
    sam2State.mats.dst = new cv.Mat(height, width, cv.CV_8UC4);
    sam2State.mats.overlay = new cv.Mat(height, width, cv.CV_8UC4);
    sam2State.mats.contoursVec = new cv.MatVector();
  } catch (e) {
    console.error("OpenCV Init Error in SAM2:", e);
  }
}

function renderSam2Frame() {
  if (sam2State.video.paused || sam2State.video.ended) {
    sam2State.reqId = requestAnimationFrame(renderSam2Frame);
    return;
  }

  try {
    let width = sam2State.video.videoWidth;
    let height = sam2State.video.videoHeight;

    if (width === 0 || height === 0) {
      requestAnimationFrame(renderSam2Frame);
      return;
    }

    // Draw video frame to canvas first (using 2D context) to get pixel data for OpenCV
    // Note: OpenCV.js cap.read(video) is cleaner but we can also drawImage -> matFromImageData
    // Let's use the cap approach if we created a cap, but here we'll use a hidden approach:
    // Draw video to a temporary canvas or directly to mat?
    // Easiest approach for syncing: Draw video to canvas, then process canvas.

    sam2State.ctx.drawImage(sam2State.video, 0, 0, width, height);

    // Read image data from canvas into Mat
    let imageData = sam2State.ctx.getImageData(0, 0, width, height);
    sam2State.mats.src.data.set(imageData.data);

    // --- PROCESSING ---
    // 1. Calculate Frame Index
    let currentTime = sam2State.video.currentTime;
    let frameIdx = Math.floor(currentTime * sam2State.fps);

    // 2. Get Contours
    let frameContours = sam2State.data.frames[frameIdx];

    if (frameContours) {
      // Clean up previous vector
      sam2State.mats.contoursVec.delete();
      sam2State.mats.contoursVec = new cv.MatVector();

      let tempMats = [];

      for (let points of frameContours) {
        let flatPoints = points.flat();
        let cntMat = cv.matFromArray(points.length, 1, cv.CV_32SC2, flatPoints);
        sam2State.mats.contoursVec.push_back(cntMat);
        tempMats.push(cntMat);
      }

      // Create overlay
      sam2State.mats.src.copyTo(sam2State.mats.overlay);

      // Draw filled contours
      let colorFill = new cv.Scalar(255, 0, 0, 100); // R, G, B, Alpha
      let colorLine = new cv.Scalar(255, 0, 0, 255);

      // Note: OpenCV JS drawContours modifies the image directly
      cv.drawContours(
        sam2State.mats.overlay,
        sam2State.mats.contoursVec,
        -1,
        colorFill,
        -1,
      );

      // Blend
      cv.addWeighted(
        sam2State.mats.overlay,
        0.4,
        sam2State.mats.src,
        0.6,
        0,
        sam2State.mats.dst,
      );

      // Draw Outline
      cv.drawContours(
        sam2State.mats.dst,
        sam2State.mats.contoursVec,
        -1,
        colorLine,
        2,
      );

      // Cleanup temp mats
      tempMats.forEach((m) => m.delete());
    } else {
      // No contours, just show original
      sam2State.mats.src.copyTo(sam2State.mats.dst);
    }

    // --- RENDER ---
    cv.imshow("sam2-canvas", sam2State.mats.dst);
  } catch (e) {
    console.error("SAM2 Render Error:", e);
  }

  sam2State.reqId = requestAnimationFrame(renderSam2Frame);
}
