# Adobe Content Intelligence Simulator

**Inter IIT Tech Meet Madras Challenge | GDG Club, IIT Indore**

A closed-loop AI system for social media marketing that combines **behavior prediction** and **content generation** to optimize engagement on enterprise Twitter accounts.

## 🎯 Overview

This project tackles two interconnected challenges:
- **Task 1 (Behavior Simulation)**: Predict the number of likes a tweet will receive based on multimodal context (text, images, metadata)
- **Task 2 (Content Simulation)**: Generate high-quality, engagement-optimized tweet text conditioned on visual media and target engagement metrics

## 📊 Key Components

### Task 1: Engagement Prediction
- **Baseline Approach**: Text + metadata regression (Ridge/XGBoost) → R² = 0.17
- **Final Approach**: Multimodal fusion using DistilBERT embeddings + BLIP-2 visual embeddings with log-transformed targets
- **Challenge**: Power-law distribution of engagement data; addressed through log-transformation

### Task 2: Tweet Generation  
- **Baseline**: Mistral-7B-Instruct without visual grounding → generic outputs
- **Final Pipeline**: 
  - Florence-2 for high-quality image/video captioning
  - Entropy-based keyframe selection for video posts
  - LoRA fine-tuned Mistral-7B-Instruct v0.2
  - Post-processing for artifact removal
- **Results**: Perplexity reduced from 10.12 → 3.63; ROUGE-4 = 0.57

## 👥 Team

| Name | Role |
|------|------|
| **Ketki Patil** | Task 1 Implementation |
| **Sanskriti Jain** | Task 1 Implementation |
| **Shlok Parikh** | Task 2 Implementation |
| **Yogendra Singh** | Task 2 Implementation |

## 🛠️ Tech Stack

- **Models**: DistilBERT, BLIP-2, Florence-2, Mistral-7B-Instruct, XGBoost, LightGBM
- **Frameworks**: PyTorch, Transformers, LoRA
- **Data Processing**: Pandas, NumPy
- **Evaluation**: RMSE, ROUGE, BLEU, CIDER

## 📈 Key Findings

1. **Visual context is indispensable** for engagement prediction; text-only models fail (R² = 0.17)
2. **Log-transformation critical** for handling skewed engagement distributions (mean = 718, median = 73)
3. **Vision-Language grounding essential** for meaningful tweet generation
4. **Data quality matters**: Broken media URLs significantly limited full-scale training

## 🔮 Future Directions

- Train multimodal regression on full 300K dataset (currently limited to 3K due to broken URLs)
- Implement temporal-aware video embeddings
- Build self-improving feedback loop: Generate → Predict → Reinforce
- Add explainability layer (SHAP/LIME) for actionable marketing insights

## 📝 Dataset

- **Size**: ~300K enterprise tweets (2018-2023)
- **Features**: Date, company, username, timestamp, tweet text, media URLs, likes
- **Source**: Adobe Experience Cloud challenge data

---

*This project validates a multimodal pipeline for both forecasting social media engagement and creating optimized brand-consistent content, offering a strong baseline for AI-assisted marketing tools.*
