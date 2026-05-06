# Computer Vision Project – Explainability, Fidelity & Stability

## Authors
- Frederik
- Mathijs Tobé
- Giulio Lo Cigno  

## Course
This project is developed for the **Master’s course in Computer Vision** at **Aarhus University**.

---

## Overview

The goal of this project is to **analyze and improve interpretability metrics**, specifically **fidelity** and **stability**, for deep learning models applied to image classification tasks.

We focus on evaluating how reliable and consistent different explanation methods are when applied to Convolutional Neural Networks (CNNs). The project combines:
- Model training and experimentation
- Explainability method implementation and analysis
- Systematic evaluation across configurations

---

## Project Evolution

### Initial Setup (CIFAKE)

We initially reproduced the CNN architecture described in:

> *“CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images”*

The model was trained on the **CIFAKE dataset**, with the aim of reproducing both classification performance and interpretability evaluation.

### Dataset Shift

During experimentation, we identified a methodological issue:

- The **CIFAKE task (real vs AI-generated images)** is inherently ambiguous in terms of *localized explanations*.
- Metrics like **fidelity** and **stability** assume that explanations correspond to meaningful, spatially grounded features.
- In CIFAKE, the discriminative features are often **distributed or non-semantic**, making evaluation unreliable.

Because of this conflict, we switched to the **Cats vs Dogs dataset**, where:

- The classification task is **visually grounded**
- Saliency maps can be meaningfully interpreted
- Fidelity and stability metrics are more aligned with the problem structure

---

## Methods

### CNN Model

- The baseline CNN is adapted from the CIFAKE paper
- Multiple variants were trained by modifying **hyperparameters** (e.g., depth, learning rate, regularization)

---

### Explainability Methods

We evaluated two main approaches:

#### 1. SHAP (DeepExplainer)
- Used `shap.DeepExplainer()` for model interpretability
- Provides **pixel-level importance values**
- Allows quantitative comparison via fidelity and stability

#### 2. Grad-CAM (Custom Implementation)
- Implemented Grad-CAM from scratch
- Produces **class-discriminative heatmaps**
- Useful for qualitative and visual inspection

---

### Explored but Discarded

#### Counterfactual Explanations
We briefly explored generating counterfactuals, but discarded this direction due to:
- High computational cost
- Difficulty defining meaningful perturbations for images
- Misalignment with the main evaluation metrics (fidelity & stability)

---

## Evaluation Pipeline

We developed a **systematic evaluation framework** that:

1. Trains multiple CNN variants with different hyperparameters
2. Applies SHAP and Grad-CAM to each model
3. Computes:
   - **Fidelity** (how well explanations reflect model behavior)
   - **Stability** (sensitivity to small input perturbations)
4. Generates visual outputs:
   - Saliency maps
   - Pixel importance overlays

This allows direct comparison across:
- Model configurations
- Explainability methods

---

## Outputs

The project produces:

- Quantitative evaluation results (fidelity & stability)
- Visualizations of explanations:
  - Heatmaps (Grad-CAM)
  - Pixel attribution maps (SHAP)
- Comparative analysis across model variants

