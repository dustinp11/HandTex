const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let isDrawing = false;

canvas.width = 800;
canvas.height = 800;

function startDrawing(e) {
    isDrawing = true;
}

function draw(e) {
    if (!isDrawing) return;
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
}

async function predict() {
}

canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);

document.getElementById('clearBtn').addEventListener('click', clearCanvas);
document.getElementById('predictBtn').addEventListener('click', predict);