const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const result = document.getElementById("result");
let timeout = null;

// Set black background
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// White drawing color 
ctx.strokeStyle = "white";
ctx.lineWidth = 15;
ctx.lineCap = "round";

let drawing = false;

// Mouse events
canvas.addEventListener("mousedown", () => drawing = true);

canvas.addEventListener("mouseup", () => {
    drawing = false;
    ctx.beginPath();

    if (timeout) clearTimeout(timeout);

    timeout = setTimeout(() => {
        sendForPrediction();
    }, 300);
});

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

// Clear Button
clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    result.innerText = "-";
});

// Live Prediction
function sendForPrediction(){
    const imageData = canvas.toDataURL("image/png");
    
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json"},
        body: JSON.stringify({ image: imageData})
    })
    .then(res => res.json())
    .then(data => {
        result.innerText = data.prediction;
    });
}


// What did does? Right now, you can draw on canvas, clicking predict sends dummy data to /predict, backend returns { prediction : 5}, Result displays
// No ML yet, just testing pipeline. 
// const ctx = canvas.getContent("2d");
// This is very imp, Canvas itself is just a blank rectangle, getContext("2d") gives you the drawing tool
// canvas = paper, ctx = pen, all drawing happens through ctx

// Set black background
// ctx.fillStyle = "black";
// ctx.fillRect(0, 0, canvas.width, canvas.height);
// fillStyle sets fill color, fillRect(x, y, width, height) draws rectangle
// This paints entire canvas black, Why? MNIST digits are white on black background

// White drawing color 
//ctx.strokeStyle = "white";
//ctx.lineWidth = 15;
//ctx.lineCap = "round";
// It is used to set drawing color (white here)
// strokeStyle is for line color, lineWidth is thickness, lineCap = "round" which gives smooth edges
// Now drawing will be white thick strokes

// Concept for prediction logic
// Debouncing, means wait until user stops drawing for a short time, then send prediction request
