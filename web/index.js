import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";
env.allowLocalModels = false;

/****************** Kamera **************************/
// Reference to video element.
var video = document.querySelector("#video");

// Ensure cross-browser functionality.
navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(stream => video.srcObject = stream)
    .catch(e => document.querySelector('#camera').innerHTML = "<p>Kamera nicht benutzbar!</p>");

/******************************************************/

var camera_button = document.querySelector('#camera_button')
camera_button.addEventListener(
    "click",
    (ev) => {
        const video = document.getElementById('video');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageContainer = document.getElementById("image-container");
        imageContainer.innerHTML = "";
        const image = document.createElement("img");
        image.src = canvas.toDataURL('image/png');
        imageContainer.appendChild(image);
        detect(image);  // Uncomment this line to run the model
        ev.preventDefault();
    },
    false,
);

function resizeImage(image, width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, width, height);
    return canvas;
}

/****************** Ai Stuff **************************/

let capturedImage = document.getElementById("video");
const imageContainer = document.getElementById("image-container");
const status = document.getElementById("status");
const result = document.getElementById("result");
status.textContent = "Loading model...";
const detector = await pipeline("object-detection", "Xenova/detr-resnet-50");
status.textContent = "Ready";

async function detect(img) {
    status.textContent = "Analysing...";
    const output = await detector(img.src, {
        threshold: 0.7,
        percentage: true,
    });
    status.textContent = "";
    console.log("output", output);
    result.textContent = "";
   output.forEach(item => {
    result.textContent += item.label + " " + item.score + "";
});
}
