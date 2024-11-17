import {AutoModel, AutoProcessor, RawImage} from "https://cdn.jsdelivr.net/npm/@xenova/transformers";

/****************** Kamera **************************/
// Reference to video element.
var video = document.querySelector("#video");

const videoConstraints = {
    video: { facingMode: 'environment' }
    }

// Ensure cross-browser functionality
navigator.mediaDevices.getUserMedia(videoConstraints)
    .then(stream => video.srcObject = stream)
    .catch(e => document.querySelector('#camera').innerHTML = "<p>Kamera nicht benutzbar!</p>");

// Change Cameras
const cameraOptions = document.querySelector('.video-options>select');
const getCameraSelection = async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    const options = videoDevices.map(videoDevice => {
        return `<option value="${videoDevice.deviceId}">${videoDevice.label}</option>`;
    });
    cameraOptions.innerHTML = options.join('');
};
getCameraSelection();

var camera_change_button = document.querySelector('#camera_dreh_button')
camera_change_button.addEventListener(
    "click",
    (ev) => {
        if ('mediaDevices' in navigator && navigator.mediaDevices.getUserMedia) {
            const updatedConstraints = {
                ...videoConstraints,
                deviceId: {
                    exact: cameraOptions.value
                }
            };
            navigator.mediaDevices.getUserMedia(updatedConstraints).then(stream => video.srcObject = stream);
        }
        ev.preventDefault();
    },
    false,
);


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

const status = document.getElementById("status");
const result = document.getElementById("result");
status.textContent = "Loading model...";
const model = await AutoModel.from_pretrained('onnx-community/yolov10n', {
    // quantized: false,    // (Optional) Use unquantized version.
})
const processor = await AutoProcessor.from_pretrained('onnx-community/yolov10n');
status.textContent = "Ready";
setTimeout(autoDetect, 1000);       //hacky, ToDo only after camera feed is loaded

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
