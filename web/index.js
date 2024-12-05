import {AutoModel, AutoProcessor, RawImage} from "./libs/xenova_transformers.js";
import { pipeline } from "./libs/huggingface_transformers.js";
import { getTranslation, language } from './localisation.js';

/****************** Constants **************************/
let audioModelLoaded = false;
let objectDetectionModelLoaded = false;
let depthModelLoaded = false;
let cameraLoaded = false;

let audioOutput = false;
let objectDetection = false;
let depthDetection = false;

const status = document.getElementById("status");
const result = document.getElementById("result");

const buttonStandardColor = "gray";
const buttonSuccessColor = "green";

let lastOutput = "";

let cameraFacingUser = false;

/****************** Kamera **************************/

let video = document.querySelector("#video");   // Reference to the video element.

let videoConstraints = {
    video: { facingMode: 'environment' }
}

refreshCamera();
async function refreshCamera() {
    navigator.mediaDevices.getUserMedia(videoConstraints)
        .then(stream => {
            video.srcObject = stream;
            cameraLoaded = true;
        })
        .catch(e => {
            document.querySelector('#camera').innerHTML = "<p>Kamera nicht benutzbar!</p>";
            cameraLoaded = false;
        });
}

/****************** Buttons **************************/

let camera_change_button = document.querySelector('#camera_dreh_button')
camera_change_button.addEventListener(
    "click",
    (ev) => {
        if (cameraFacingUser) {
            videoConstraints.video.facingMode = 'environment';
            cameraFacingUser = false;
        } else {
            videoConstraints.video.facingMode = 'user';
            cameraFacingUser = true;
        }
        refreshCamera();
        ev.preventDefault();
    },
    false,
);

/*
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
*/

let audio_button = document.querySelector('#toggleAudioOut')
audio_button.addEventListener(
    "click",
    (ev) => {
        if (audioOutput) {
            audioOutput = false;
            audio_button.style.backgroundColor = buttonStandardColor;
        } else {
            audioOutput = true;
            if (!audioModelLoaded) {
                loadAudioModel();
            } else {
                audio_button.style.backgroundColor = buttonSuccessColor;
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
            objectDetection_button.style.backgroundColor = buttonStandardColor;
        } else {
            objectDetection = true;
            if (!objectDetectionModelLoaded) {
                loadObjectModel();
            } else {
                objectDetection_button.style.backgroundColor = buttonSuccessColor;
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
            depthDetection_button.style.backgroundColor = buttonStandardColor;
        } else {
            depthDetection = true;
            if (!depthModelLoaded) {
                loadDepthModel();
            } else {
                depthDetection_button.style.backgroundColor = buttonSuccessColor;
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
setStatus(true, "");
setTimeout(autoDetect, 2000);       //hacky, ToDo only after camera feed is loaded

async function loadAudioModel() {
    setStatus(false, "audio_output");
    synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', { quantized: false });
    speaker_embeddings = './libs/speaker_embeddings.bin';
    audioModelLoaded = true;
    setStatus(true, "");
    audio_button.style.backgroundColor = buttonSuccessColor;
    audioOutput = true;
}

async function loadObjectModel() {
    setStatus(false, "object_dect");
    objectDectModel = await AutoModel.from_pretrained('onnx-community/yolov10n', {
        // quantized: false,    // (Optional) Use unquantized version.
    })
    objectDectProcessor = await AutoProcessor.from_pretrained('onnx-community/yolov10n');
    objectDetectionModelLoaded = true;
    setStatus(true, "");
    objectDetection_button.style.backgroundColor = buttonSuccessColor;
    objectDetection = true;
}

async function loadDepthModel() {       // Create depth-estimation pipeline
    setStatus(false, "depth_dect");
    depth_estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small');
    depthModelLoaded = true;
    setStatus(true, "");
    depthDetection_button.style.backgroundColor = buttonSuccessColor;
    depthDetection = true;
}

/****************** Ai Stuff **************************/

async function playTextToSpeech(text) {
    if (!audioModelLoaded) {
        return;
    }
    const audioResult = await synthesizer(text, { speaker_embeddings });
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    // Create an audio buffer
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
        console.log("Camera not loaded");
        setTimeout(autoDetect, 1000);
        return;
    }
    const image = await getImageFromCamera();
    const rawImage = await RawImage.read(image.src);
    if (objectDetection) {
        try {
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
        } catch (error) {
            console.error(error);
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

/****************** Output Stuff **************************/


function addTextToOutput(text) {
    if (text !== lastOutput) {
        result.textContent = result.textContent + text + " | ";
        result.scrollTop = result.scrollHeight;     // Auto-scroll to the bottom
        lastOutput = text;
    }
    if (audioOutput) {
        playTextToSpeech(text);
    }
}

function setStatus(ready, model) {
    model = getTranslation(model);
    if (language === "en") {
        status.textContent = ready ? "Ready" : "Loading " + model + "...";
    } else if(language === "de") {
        status.textContent = ready ? "Bereit" : model + " wird geladen...";
    }
}
