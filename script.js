let model;
let classNames = ["No Flood", "Flood"];

const imageUpload = document.getElementById("imageUpload");
const previewImage = document.getElementById("previewImage");
const resultEl = document.getElementById("result");
const confidenceEl = document.getElementById("confidence");

async function loadClassNames() {
  try {
    const response = await fetch("./tfjs_model/class_names.json");
    if (response.ok) {
      classNames = await response.json();
      console.log("Class names loaded:", classNames);
    }
  } catch (error) {
    console.warn("class_names.json not loaded, using default class names.");
  }
}

async function loadModel() {
  resultEl.innerText = "Loading model...";
  confidenceEl.innerText = "Please wait until the model is ready.";

  try {
    await loadClassNames();

    model = await tf.loadLayersModel("./tfjs_model/model.json");

    console.log("Model loaded successfully");
    resultEl.innerText = "Model loaded successfully.";
    confidenceEl.innerText = "Upload an image and click Predict Flood Status.";
  } catch (error) {
    console.error("Model load error:", error);
    resultEl.innerText = "Model failed to load.";
    confidenceEl.innerText = error.message;
  }
}

loadModel();

imageUpload.addEventListener("change", function (event) {
  const file = event.target.files[0];

  if (file) {
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";

    resultEl.innerText = "Image uploaded.";
    confidenceEl.innerText = model
      ? "Click Predict Flood Status."
      : "Waiting for model to load.";
  }
});

async function predictImage() {
  if (!model) {
    resultEl.innerText = "Model is not ready.";
    confidenceEl.innerText = "Please wait until the model is loaded.";
    return;
  }

  if (!previewImage.src) {
    resultEl.innerText = "Please upload an image first.";
    confidenceEl.innerText = "";
    return;
  }

  resultEl.innerText = "Predicting...";
  confidenceEl.innerText = "";

  try {
    const tensor = tf.browser.fromPixels(previewImage)
      .resizeBilinear([224, 224])
      .expandDims(0)
      .toFloat()
      .div(255.0);

    const prediction = model.predict(tensor);
    const predictionData = await prediction.data();

    let predictedIndex;
    let confidence;

    if (predictionData.length === 1) {
      const probability = predictionData[0];
      predictedIndex = probability >= 0.5 ? 1 : 0;
      confidence = predictedIndex === 1 ? probability : 1 - probability;
    } else {
      confidence = Math.max(...predictionData);
      predictedIndex = predictionData.indexOf(confidence);
    }

    const predictedLabel = classNames[predictedIndex];

    resultEl.innerText = "Prediction: " + predictedLabel;
    confidenceEl.innerText = "Confidence: " + (confidence * 100).toFixed(2) + "%";

    tensor.dispose();
    prediction.dispose();
  } catch (error) {
    console.error("Prediction error:", error);
    resultEl.innerText = "Prediction failed.";
    confidenceEl.innerText = error.message;
  }
}
