// onnx-inference.js (升級版)
const CONF_THRESHOLD = 0.4;
const IOU_THRESHOLD = 0.45;

async function runYOLO(session, video) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const countText = document.getElementById('count');

  async function detect() {
    const inputTensor = preprocess(video);
    const results = await session.run({ images: inputTensor });
    const boxes = postprocess(results);

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 畫藥錠中心點
    boxes.forEach(b => {
      const cx = (b[0] + b[2]) / 2;
      const cy = (b[1] + b[3]) / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'lime';
      ctx.fill();
    });

    // 顯示總數
    countText.innerText = "Total: " + boxes.length;

    requestAnimationFrame(detect);
  }
  detect();
}

function preprocess(video) {
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = 640;
  tmpCanvas.height = 640;
  const tmpCtx = tmpCanvas.getContext('2d');
  tmpCtx.drawImage(video, 0, 0, 640, 640);
  const imgData = tmpCtx.getImageData(0, 0, 640, 640);

  const data = new Float32Array(640 * 640 * 3);
  for (let i = 0; i < imgData.data.length; i += 4) {
    const j = (i / 4) * 3;
    data[j] = imgData.data[i] / 255;
    data[j + 1] = imgData.data[i + 1] / 255;
    data[j + 2] = imgData.data[i + 2] / 255;
  }
  return new ort.Tensor('float32', data, [1, 3, 640, 640]);
}

function postprocess(results) {
  const output = results[Object.keys(results)[0]].data;
  const numClasses = 1; // 只有一個類別：藥錠
  const boxes = [];

  for (let i = 0; i < output.length; i += (5 + numClasses)) {
    const x = output[i];
    const y = output[i + 1];
    const w = output[i + 2];
    const h = output[i + 3];
    const conf = output[i + 4];
    if (conf < CONF_THRESHOLD) continue;

    const x1 = (x - w / 2);
    const y1 = (y - h / 2);
    const x2 = (x + w / 2);
    const y2 = (y + h / 2);
    boxes.push([x1, y1, x2, y2, conf]);
  }

  return nonMaxSuppression(boxes, IOU_THRESHOLD);
}

function nonMaxSuppression(boxes, iouThreshold) {
  boxes.sort((a, b) => b[4] - a[4]);
  const selected = [];

  while (boxes.length > 0) {
    const chosen = boxes.shift();
    selected.push(chosen);
    boxes = boxes.filter(b => iou(chosen, b) < iouThreshold);
  }
  return selected;
}

function iou(box1, box2) {
  const x1 = Math.max(box1[0], box2[0]);
  const y1 = Math.max(box1[1], box2[1]);
  const x2 = Math.min(box1[2], box2[2]);
  const y2 = Math.min(box1[3], box2[3]);
  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
  const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
  return interArea / (box1Area + box2Area - interArea);
}
