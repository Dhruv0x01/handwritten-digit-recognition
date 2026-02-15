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
ctx.lineWidth = 10;
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




