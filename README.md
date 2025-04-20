<h2 style="margin-top: 1em;">Project Overview</h2>
<p>&nbsp;&nbsp;A lightweight pipeline for agricultural image classification that combines classic and deep learning features.</p>

<ul>
  <li>Extracts handcrafted descriptors:
    <ul>
      <li>&nbsp;&nbsp;Local Binary Patterns (LBP)</li>
      <li>&nbsp;&nbsp;Gray‑Level Co‑occurrence Matrix (GLCM)</li>
      <li>&nbsp;&nbsp;Gabor filter features</li>
      <li>&nbsp;&nbsp;ORB features with a Bag‑of‑Visual‑Words (BoVW) model</li>
    </ul>
  </li>
  <li>Augments these with deep embeddings from PyTorch’s pretrained VGG16</li>
  <li>Performs feature selection and dimensionality reduction using PCA and RFE</li>
  <li>Trains an SVM classifier via scikit‑learn for final predictions</li>
</ul>
