import { AutoModel, AutoProcessor, RawImage } from "https://cdn.jsdelivr.net/npm/@xenova/transformers";

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
        //detect(image);  // Uncomment this line to run the model
        image.onload = () => detect(image);
        ev.preventDefault();
    },
    false,
);

/****************** Ai Stuff **************************/

let capturedImage = document.getElementById("video");
const imageContainer = document.getElementById("image-container");
const status = document.getElementById("status");
const result = document.getElementById("result");
status.textContent = "Loading model...";
const model = await AutoModel.from_pretrained('onnx-community/yolov10n', {
    // quantized: false,    // (Optional) Use unquantized version.
})
const processor = await AutoProcessor.from_pretrained('onnx-community/yolov10n');
status.textContent = "Ready";
autoDetect();

async function detect(imageElement) {
    // Read image and run processor
    const image = await RawImage.read(imageElement.src);
    const { pixel_values, reshaped_input_sizes } = await processor(image);
    // Run object detection
    const { output0 } = await model({ images: pixel_values });
    const predictions = output0.tolist()[0];
    const threshold = 0.6;
    const [newHeight, newWidth] = reshaped_input_sizes[0]; // Reshaped height and width
    const [xs, ys] = [image.width / newWidth, image.height / newHeight]; // x and y resize scales
    for (const [xmin, ymin, xmax, ymax, score, id] of predictions) {
        if (score < threshold) continue;
        // Convert to original image coordinates
        const bbox = [xmin * xs, ymin * ys, xmax * xs, ymax * ys].map(x => x.toFixed(2)).join(', ');
        console.log(`Found "${model.config.id2label[id]}" at [${bbox}] with score ${score.toFixed(2)}.`);
        result.textContent = model.config.id2label[id];
    }
}


async function autoDetect() {
    const video = document.getElementById('video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = document.createElement("img");
    image.src = canvas.toDataURL('image/png');
    image.onload = async () => {
        // Read image and run processor
        const rawImage = await RawImage.read(image.src);
        const {pixel_values, reshaped_input_sizes} = await processor(rawImage);
        // Run object detection
        const {output0} = await model({images: pixel_values});
        const predictions = output0.tolist()[0];
        const threshold = 0.6;
        const [newHeight, newWidth] = reshaped_input_sizes[0]; // Reshaped height and width
        const [xs, ys] = [rawImage.width / newWidth, rawImage.height / newHeight]; // x and y resize scales
        for (const [xmin, ymin, xmax, ymax, score, id] of predictions) {
            if (score < threshold) continue;
            // Convert to original image coordinates
            const bbox = [xmin * xs, ymin * ys, xmax * xs, ymax * ys].map(x => x.toFixed(2)).join(', ');
            console.log(`Found "${model.config.id2label[id]}" at [${bbox}] with score ${score.toFixed(2)}.`);
            result.textContent = result.textContent + "\n" + model.config.id2label[id];
        }
        setTimeout(autoDetect, 100);
    }
}


