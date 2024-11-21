import {AutoModel, AutoProcessor, RawImage} from "https://cdn.jsdelivr.net/npm/@xenova/transformers";
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers";

/****************** Constants **************************/

let audioModelLoaded = false;
let objectDetectionModelLoaded = false;
let depthModelLoaded = false;
let cameraLoaded = true;    //FixMe

let audioOutput = false;
let objectDetection = false;
let depthDetection = false;

const status = document.getElementById("status");
const result = document.getElementById("result");

/****************** Kamera **************************/

let video = document.querySelector("#video");   // Reference to the video element.

const videoConstraints = {
    video: { facingMode: 'environment' }
}

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


/****************** Buttons **************************/

let camera_change_button = document.querySelector('#camera_dreh_button')
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

let camera_button = document.querySelector('#camera_button')
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
        image.onload = () => detect(image);
        ev.preventDefault();
    },
    false,
);

let audio_button = document.querySelector('#toggleAudioOut')
audio_button.addEventListener(
    "click",
    (ev) => {
        if (audioOutput) {
            audioOutput = false;
            audio_button.style.backgroundColor = "gray";
        } else {
            audioOutput = true;
            if (!audioModelLoaded) {
                loadAudioModel();
            } else {
                audio_button.style.backgroundColor = "green";
            }
        }
        //ev.preventDefault();
    },
    false,
);

let objectDetection_button = document.querySelector('#toggleObjectDetection')
objectDetection_button.addEventListener(
    "click",
    (ev) => {
        if (objectDetection) {
            objectDetection = false;
            objectDetection_button.style.backgroundColor = "gray";
        } else {
            objectDetection = true;
            if (!objectDetectionModelLoaded) {
                loadObjectModel();
            } else {
                objectDetection_button.style.backgroundColor = "green";
            }
        }
        //ev.preventDefault();
    },
    false,
);

let depthDetection_button = document.querySelector('#toggleDepthDetection')
depthDetection_button.addEventListener(
    "click",
    (ev) => {
        if (depthDetection) {
            depthDetection = false;
            depthDetection_button.style.backgroundColor = "gray";
        } else {
            depthDetection = true;
            if (!depthModelLoaded) {
                loadDepthModel();
            } else {
                depthDetection_button.style.backgroundColor = "green";
            }
        }
        //ev.preventDefault();
    },
    false,
);

/****************** Ai Setup **************************/

let objectDectModel;
let objectDectProcessor;
let depth_estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small');
// TTS Pipeline
let synthesizer;
let speaker_embeddings;
status.textContent = "Ready";
setTimeout(autoDetect, 2000);       //hacky, ToDo only after camera feed is loaded

async function loadAudioModel() {
    status.textContent = "Audio Model Loading...";
    synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', { quantized: false });
    speaker_embeddings = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin';
    audioModelLoaded = true;
    status.textContent = "Ready";
    audio_button.style.backgroundColor = "green";
    audioOutput = true;
}

async function loadObjectModel() {
    status.textContent = "Object Detection Model Loading...";
    objectDectModel = await AutoModel.from_pretrained('onnx-community/yolov10n', {
        // quantized: false,    // (Optional) Use unquantized version.
    })
    objectDectProcessor = await AutoProcessor.from_pretrained('onnx-community/yolov10n');
    objectDetectionModelLoaded = true;
    status.textContent = "Ready";
    objectDetection_button.style.backgroundColor = "green";
    objectDetection = true;
}

async function loadDepthModel() {       // Create depth-estimation pipeline
    status.textContent = "Depth Detection Model Loading...";
    depth_estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small');
    depthModelLoaded = true;
    status.textContent = "Ready";
    depthDetection_button.style.backgroundColor = "green";
    depthDetection = true;
}

/****************** Ai Stuff **************************/


async function detect(imageElement) {
    // Read image and run processor
    const image = await RawImage.read(imageElement.src);
    // Run depth detection
    const { depth } = await depth_estimator(imageElement.src);
    const { pixel_values, reshaped_input_sizes } = await objectDectProcessor(image);  //ToDo wait for model to load
    // Run object detection
    const { output0 } = await objectDectModel({ images: pixel_values });
    const predictions = output0.tolist()[0];
    const threshold = 0.6;
    const [newHeight, newWidth] = reshaped_input_sizes[0]; // Reshaped height and width
    const [xs, ys] = [image.width / newWidth, image.height / newHeight]; // x and y resize scales
    for (const [xmin, ymin, xmax, ymax, score, id] of predictions) {
        if (score < threshold) continue;
        // Convert to original image coordinates
        const xMiddle = (xmax * xs) - (xmin * xs);
        const yMiddle = (xmax * xs) - (ymin * ys);
        const onedCoord = Math.round(xMiddle * yMiddle);
        const middleOfPicture = (image.width * image.height) / 2;
        const bbox = [xmin * xs, ymin * ys, xmax * xs, ymax * ys].map(x => x.toFixed(2)).join(', ');
        console.log(`Found "${objectDectModel.config.id2label[id]}" at [${bbox}] with score ${score.toFixed(2)}.`);
        console.log(depth.data[middleOfPicture]);
        const newText = objectDectModel.config.id2label[id] +" depth: " + depth.data[middleOfPicture] + " | ";
        result.textContent = result.textContent + newText;

        const audioResult = await synthesizer(newText, { speaker_embeddings });
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        // Create audio buffer
        const audioBuffer = audioContext.createBuffer(
            1, // mono
            audioResult.audio.length,
            audioResult.sampling_rate
        );
        // Copy audio data to buffer
        audioBuffer.copyToChannel(audioResult.audio, 0);
        // Create audio source
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        // Play audio
        source.start(0);
    }
}

async function playTextToSpeech(text) {
    if (!audioModelLoaded) {
        return;
    }
    const audioResult = await synthesizer(text, { speaker_embeddings });
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    // Create audio buffer
    const audioBuffer = audioContext.createBuffer(
        1, // mono
        audioResult.audio.length,
        audioResult.sampling_rate
    );
    // Copy audio data to buffer
    audioBuffer.copyToChannel(audioResult.audio, 0);
    // Create audio source
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    // Play audio
    source.start(0);
}

function getImageFromCamera() {
    return new Promise((resolve) => {
        const video = document.getElementById('video');
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const image = document.createElement("img");
        image.src = canvas.toDataURL('image/png');
        image.onload = () => {
            resolve(image);
        };
    });
}

async function autoDetect() {
    if (!cameraLoaded) {
        setTimeout(autoDetect, 1000);
    }
    const image = await getImageFromCamera();
    const rawImage = await RawImage.read(image.src);
    if (objectDetection) {
        const {pixel_values, reshaped_input_sizes} = await objectDectProcessor(rawImage);
        // Run object detection
        const {output0} = await objectDectModel({images: pixel_values});
        const predictions = output0.tolist()[0];
        const threshold = 0.6;
        const [newHeight, newWidth] = reshaped_input_sizes[0]; // Reshaped height and width
        const [xs, ys] = [rawImage.width / newWidth, rawImage.height / newHeight]; // x and y resize scales
        for (const [xmin, ymin, xmax, ymax, score, id] of predictions) {
            if (score < threshold) continue;
            const bbox = [xmin * xs, ymin * ys, xmax * xs, ymax * ys].map(x => x.toFixed(2)).join(', '); // Convert to original image coordinates
            console.log(`Found "${objectDectModel.config.id2label[id]}" at [${bbox}] with score ${score.toFixed(2)}.`);
            addTextToOutput(objectDectModel.config.id2label[id]);
        }
    }
    if (depthDetection) {
        const { depth } = await depth_estimator(image.src);
        const middleOfPicture = (image.width * image.height) / 2;
        console.log(depth.data[middleOfPicture]);
        addTextToOutput(depth.data[middleOfPicture])
    }
    setTimeout(autoDetect, 100);
}

function addTextToOutput(text) {
    result.textContent = result.textContent + text + " | ";
    if (audioOutput) {
        playTextToSpeech(text);
    }
}
