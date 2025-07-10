# 🕺 Pose Estimation with Penn Action Dataset

This project implements a lightweight human **pose estimation** model based on OpenPose architecture with a MobileNet backbone, trained and evaluated on the [Penn Action Dataset](http://dreamdragon.github.io/PennAction/).

---

## 📌 Features

- 🔍 Keypoint detection via heatmaps
- 🧠 Trained using visibility-masked MSE loss
- 🎯 Evaluated with **PCK (Percentage of Correct Keypoints)**
- 💡 Modular design for easy extension (e.g., action recognition)

---

## 🧠 Model

- Backbone: `PoseEstimationWithMobileNet`
- Architecture inspired by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Heatmaps generated per joint using 2D Gaussians
- Output: multi-channel heatmaps for each joint

---

## 🗃 Dataset

- **Penn Action Dataset** (requires manual download):  
  http://dreamdragon.github.io/PennAction/

> ⚠️ The dataset (`Penn_Action/`) is excluded from this repo due to size.  
> Please download and extract it into your project folder like so:

