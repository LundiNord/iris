//import * as tf from '@tensorflow/tfjs';
import {IMAGENET_CLASSES} from './imagenet_classes.js';


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
        let img = document.getElementById('video');
        if (img.videoWidth && img.videoHeight) {
            const resizedImg = resizeImage(img, IMAGE_SIZE, IMAGE_SIZE);
            predict(resizedImg);
        } else {
            console.error("Image not loaded or has incorrect dimensions.");
        }
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

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let mobilenet;
const mobilenetDemo = async () => {
    status('Loading model...');

    mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH, {fromTFHub: true});

    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

    status('');
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
    status('Predicting...');

    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    const startTime1 = performance.now();
    // The second start time excludes the extraction and preprocessing and
    // includes only the predict() call.
    let startTime2;
    const logits = tf.tidy(() => {
        // tf.browser.fromPixels() returns a Tensor from an image element.
        const img = tf.cast(tf.browser.fromPixels(imgElement), 'float32');

        const offset = tf.scalar(127.5);
        // Normalize the image from [0, 255] to [-1, 1].
        const normalized = img.sub(offset).div(offset);

        // Reshape to a single-element batch so we can pass it to predict.
        const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

        startTime2 = performance.now();
        // Make a prediction through mobilenet.
        return mobilenet.predict(batched);
    });

    // Convert logits to probabilities and class names.
    const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
    const totalTime1 = performance.now() - startTime1;
    const totalTime2 = performance.now() - startTime2;
    status(`Done in ${Math.floor(totalTime1)} ms ` +
        `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

    // Show the classes in the DOM.
    showResults(classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(logits, topK) {
    const values = await logits.data();

    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
        valuesAndIndices.push({value: values[i], index: i});
    }
    valuesAndIndices.sort((a, b) => {
        return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
        topkValues[i] = valuesAndIndices[i].value;
        topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
        topClassesAndProbs.push({
            className: IMAGENET_CLASSES[topkIndices[i]],
            probability: topkValues[i]
        })
    }
    return topClassesAndProbs;
}

function showResults(classes) {
    const predictionContainer = document.createElement('div');
    predictionContainer.className = 'pred-container';

    const probsContainer = document.createElement('div');
    for (let i = 0; i < classes.length; i++) {
        const row = document.createElement('div');
        row.className = 'row';

        const classElement = document.createElement('div');
        classElement.className = 'cell';
        classElement.innerText = classes[i].className;
        row.appendChild(classElement);

        const probsElement = document.createElement('div');
        probsElement.className = 'cell';
        probsElement.innerText = classes[i].probability.toFixed(3);
        row.appendChild(probsElement);

        probsContainer.appendChild(row);
    }
    predictionContainer.appendChild(probsContainer);

    predictionsElement.insertBefore(
        predictionContainer, predictionsElement.firstChild);
}

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');

mobilenetDemo();
