# Adobe Content Intelligence Simulator

**Inter IIT Tech Meet Madras Challenge · GDG Club, IIT Indore**

A closed-loop AI system for social media marketing combining engagement
prediction and content generation to optimize enterprise Twitter posts.

## Overview

Two interconnected tasks:
- **Task 1 (Behavior Simulation)** — predict how many likes a tweet will
  receive from multimodal context (text, images, metadata).
- **Task 2 (Content Simulation)** — generate engagement-optimized tweet
  text conditioned on visual media and target engagement metrics.

## Approach

**Task 1 — Engagement prediction**
- Baseline: text + metadata regression (Ridge/XGBoost) → R² = 0.17.
- Final: multimodal fusion of DistilBERT text embeddings + BLIP-2 visual
  embeddings, with log-transformed targets to handle the power-law
  distribution of engagement data.

**Task 2 — Tweet generation**
- Baseline: Mistral-7B-Instruct with no visual grounding → generic output.
- Final pipeline: Florence-2 for image/video captioning, entropy-based
  keyframe selection for video posts, a LoRA-fine-tuned
  Mistral-7B-Instruct-v0.2, and post-processing for artifact removal.
- Result: perplexity reduced 10.12 → 3.63, ROUGE-4 = 0.57.

## Key findings

- Visual context is necessary for engagement prediction — text-only models
  fail (R² = 0.17).
- Log-transformation is critical for the skewed engagement distribution
  (mean = 718, median = 73 likes).
- Vision-language grounding is essential for meaningful tweet generation.
- Broken media URLs significantly limited full-scale training.

## Dataset

~300K enterprise tweets (2018–2023) from the Adobe Experience Cloud
challenge dataset — date, company, username, timestamp, tweet text, media
URLs, likes. Training was limited to ~13K of the 300K due to broken media
URLs. [Dataset + LoRA weights](https://drive.google.com/drive/folders/1Vd-GsBlN0Z3p8aDxxTV7iIjgpbxP1d8b?usp=sharing).

## Stack

`DistilBERT` · `BLIP-2` · `Florence-2` · `Mistral-7B-Instruct` · `LoRA` ·
`XGBoost` · `PyTorch` · `Transformers`

## Team

| Name | Role |
|---|---|
| Ketki Patil | Task 1 |
| Sanskriti Jain | Task 1 |
| Shlok Parikh | Task 2 |
| Yogendra Singh | Task 2 |

## Future direction

- Train the multimodal regressor on the full 300K dataset once broken
  media URLs are resolved.
- Temporal-aware video embeddings.
- Closed feedback loop: generate → predict → reinforce.
- SHAP/LIME explainability layer for marketing-actionable insights.
