let model;
let classNames = ["Flood", "No Flood"];

const MODEL_URL = "./tfjs_model/model.json";
const CLASS_NAMES_URL = "./tfjs_model/class_names.json";

async function loadClassNames() {
  try {
    const response = await fetch(CLASS_NAMES_URL, { cache: "no-store" });
    if (response.ok) {
      const loadedNames = await response.json();
      if (Array.isArray(loadedNames) && loadedNames.length > 0) {
        classNames = loadedNames;
      }
    }
  } catch (error) {
    console.warn("class_names.json not found, using default class names.", error);
  }
}

async function loadModel() {
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");
  const predictButton = document.getElementById("predictButton");

  resultEl.innerText = "Loading model...";
  confidenceEl.innerText = "Please wait until the model is ready.";
  predictButton.disabled = true;

  try {
    await loadClassNames();

    model = await tf.loadLayersModel(MODEL_URL);

    resultEl.innerText = "Model loaded successfully.";
    confidenceEl.innerText = "Upload an image and click Predict.";
    predictButton.disabled = false;
  } catch (error) {
    console.error("Model load error:", error);
    resultEl.innerText = "Model failed to load.";
    confidenceEl.innerText = error.message;
  }
}

function showPreview(file) {
  const preview = document.getElementById("previewImage");
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");

  if (!file) return;

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";

  resultEl.innerText = "Image uploaded.";
  confidenceEl.innerText = model ? "Click Predict to classify the image." : "Waiting for model to load.";
}

document.getElementById("imageUpload").addEventListener("change", (event) => {
  showPreview(event.target.files[0]);
});

const dropArea = document.getElementById("dropArea");

dropArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (event) => {
  event.preventDefault();
  dropArea.classList.remove("dragover");

  const file = event.dataTransfer.files[0];

  if (file) {
    document.getElementById("imageUpload").files = event.dataTransfer.files;
    showPreview(file);
  }
});

async function predictImage() {
  const image = document.getElementById("previewImage");
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");

  if (!model) {
    resultEl.innerText = "Model is not ready.";
    confidenceEl.innerText = "Please wait until the model has loaded.";
    return;
  }

  if (!image.src) {
    resultEl.innerText = "No image selected.";
    confidenceEl.innerText = "Please upload an image first.";
    return;
  }

  resultEl.innerText = "Predicting...";
  confidenceEl.innerText = "";

  try {
    const input = tf.tidy(() => {
      return tf.browser.fromPixels(image)
        .resizeBilinear([224, 224])
        .toFloat()
        .div(255.0)
        .expandDims(0);
    });

    const output = model.predict(input);
    const data = await output.data();

    input.dispose();
    output.dispose();

    let predictedIndex = 0;
    let confidence = 0;

    confidence = Math.max(...data);
    predictedIndex = data.indexOf(confidence);

    const predictedLabel = classNames[predictedIndex] || "Class " + predictedIndex;

    resultEl.innerText = "Prediction: " + predictedLabel;
    confidenceEl.innerText = "Confidence: " + (confidence * 100).toFixed(2) + "%";
  } catch (error) {
    console.error("Prediction error:", error);
    resultEl.innerText = "Prediction failed.";
    confidenceEl.innerText = error.message;
  }
}

loadModel();
