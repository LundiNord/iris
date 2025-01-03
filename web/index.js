import {AutoModel, AutoProcessor, RawImage} from "./libs/xenova_transformers.js";
import { pipeline } from "./libs/huggingface_transformers.js";
import {getAiOutputTranslation, getTranslation, language} from './localisation.js';

/****************** Constants **************************/
let audioModelLoaded = false;
let objectDetectionModelLoaded = false;
let cameraLoaded = false;

let audioOutput = false;
let objectDetection = false;
let colourDetection = false;

const status = document.getElementById("status");
const result = document.getElementById("result");

const buttonStandardColor = "gray";
const buttonSuccessColor = "green";

let lastOutput = "";

let cameraFacingUser = false;
let middleOfPictureX = 0;
let middleOfPictureY = 0;

/****************** Camera **************************/

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

let colorDetection_button = document.querySelector('#toggleColorDetection')
colorDetection_button.addEventListener(
    "click",
    (ev) => {
        if (colourDetection) {
            colourDetection = false;
            colorDetection_button.style.backgroundColor = buttonStandardColor;
            colorTrackerTask.stop();
        } else {
            colourDetection = true;
            colorDetection_button.style.backgroundColor = buttonSuccessColor;
            colorTrackerTask.run();
        }
        //ev.preventDefault();
    },
    false,
);

/****************** Color Definitions **************************/

tracking.ColorTracker.registerColor('green', function(r, g, b) {
    return r < 50 && g > 200 && b < 50;
});
tracking.ColorTracker.registerColor('red', function(r, g, b) {
    return r > 200 && g < 50 && b < 50;
});
tracking.ColorTracker.registerColor('blue', function(r, g, b) {
    return r < 50 && g < 50 && b > 200;
});
tracking.ColorTracker.registerColor('yellow', function(r, g, b) {
    return r > 200 && g > 200 && b < 50;
});
tracking.ColorTracker.registerColor('cyan', function(r, g, b) {
    return r < 50 && g > 200 && b > 200;
});
tracking.ColorTracker.registerColor('magenta', function(r, g, b) {
    return r > 200 && g < 50 && b > 200;
});
tracking.ColorTracker.registerColor('white', function(r, g, b) {
    return r > 200 && g > 200 && b > 200;
});
tracking.ColorTracker.registerColor('black', function(r, g, b) {
    return r < 50 && g < 50 && b < 50;
});

/****************** Ai Setup **************************/

let objectDectModel;
let objectDectProcessor;
let synthesizer;
let speaker_embeddings;
let colorTracker = new tracking.ColorTracker(['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'white', 'black']);
let colorTrackerTask = tracking.track('#video', colorTracker);
colorTrackerTask.stop();

setStatus(true, "");
setTimeout(autoDetect, 2000);

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
                addTextToOutput(objectDectModel.config.id2label[id],null);
            }
        } catch (error) {
            console.error(error);
        }
    }
    if (colourDetection) {
        middleOfPictureX = image.width / 2;
        middleOfPictureY = image.height / 2;
    }
    setTimeout(autoDetect, 100);
}

colorTracker.on('track', function (event) {
    //const middleOfPicture = (image.width * image.height) / 2;
    event.data.forEach(function (rect) {
        if (middleOfPictureY > rect.y && middleOfPictureY < rect.y + rect.height && middleOfPictureX > rect.x && middleOfPictureX < rect.x + rect.width) {
            console.log(rect.x, rect.y, rect.width, rect.height, rect.color);
            addTextToOutput(null, rect.color);
        }
    });
});

/****************** Output **************************/

function addTextToOutput(aiText, colorText) {
    let text = lastOutput;
    if (aiText !== null) {
        aiText = getAiOutputTranslation(aiText);
        let fields = text.split('|');
        text = aiText + " | " + fields[1];
    }
    if (colorText !== null) {
        colorText = getTranslation(colorText);
        let fields = text.split('|');
        text = fields[0] + " | " + colorText;
    }
    if (text !== lastOutput) {
        result.textContent = result.textContent + text + " | ";
        result.scrollTop = result.scrollHeight;     // Auto-scroll to the bottom
        lastOutput = text;
        if (audioOutput) {
            playTextToSpeech(text);
        }
    }
}

function setStatus(ready, model) {
    model = getTranslation(model);
    if (language === "en") {
        status.textContent = ready ? "Ready" : "Loading " + model + "...";
    } else if (language === "de") {
        status.textContent = ready ? "Bereit" : model + " wird geladen...";
    }
}
