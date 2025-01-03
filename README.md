# Iris — AI Environment Detector

![Iris Logo](iris_logo.png)


## Overview

Iris is an AI-powered environment detector that uses computer vision and audio processing to detect objects and colors in real-time. It leverages various AI models to provide accurate and efficient detection capabilities.

## Features

- **Real-time Object Detection**: Detects objects using pre-trained models.
- **Color Detection**: Identifies colors in the video feed.
- **Text-to-Speech**: Converts detected objects and colors into speech.
- **Camera Switching**: Switch between front and rear cameras.
- **Localization**: Supports multiple languages for UI and output.

## Usage

1. optional: serve /web from ordinary http server
2. Open `index.html` in your browser.
3. Allow camera access when prompted.
4. Use the buttons to toggle object detection, color detection, and audio output.

## File Structure

- `web/index.html`: Main HTML file.
- `web/index.js`: Main JavaScript file for handling logic.
- `web/localisation.js`: Handles localization.
- `web/manifest.json`: PWA manifest file.
- `web/index.css`: Main CSS file.

## Acknowledgements
- [yolov10n](https://huggingface.co/onnx-community/yolov10n) is used for object detection.
- [Xenova/speecht5_tts](https://huggingface.co/Xenova/speecht5_tts) for the text-to-speech pipeline.
- [Tracking.js](https://trackingjs.com/) for color detection.

