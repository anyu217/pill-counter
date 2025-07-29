async function runYOLO(session, video) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const countText = document.getElementById('count');

  async function detect() {
    const inputTensor = preprocess(video);
    const results = await session.run({ images: inputTensor });
    const boxes = postprocess(results);

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    boxes.forEach(b => {
      const cx = (b[0] + b[2]) / 2;
      const cy = (b[1] + b[3]) / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fillStyle = 'lime';
      ctx.fill();
    });
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
    const j = i / 4 * 3;
    data[j] = imgData.data[i] / 255;
    data[j + 1] = imgData.data[i + 1] / 255;
    data[j + 2] = imgData.data[i + 2] / 255;
  }
  return new ort.Tensor('float32', data, [1, 3, 640, 640]);
}

function postprocess(results) {
  const output = results[Object.keys(results)[0]].data;
  const boxes = [];
  for (let i = 0; i < output.length; i += 6) {
    const [x1, y1, x2, y2, conf, cls] = output.slice(i, i + 6);
    if (conf > 0.5) boxes.push([x1, y1, x2, y2]);
  }
  return boxes;
}