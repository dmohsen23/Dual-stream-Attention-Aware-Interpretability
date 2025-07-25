# Multimodal-Attention-Aware-Interpretability
_A deep learning framework for interpretable diagnosis of distal myopathy via multimodal attention-aware fusion._

---

## 📝 Overview

**Distal myopathies** are genetically heterogeneous muscle disorders characterized by specific myofiber alterations.  
This repository implements a **Multimodal Attention-Aware Fusion** model that:  
- Fuses **global** (ResNet50) + **local** (BagNet33) contextual information
- Uses the **Attention Gate** mechanism to efficiently fuse global and local contextual information.
- Generates saliency maps for interpretability  
- Evaluates interpretability using functionally grounded approaches: coherence score and incremental deletion.

---

## ✨ Features

- 🔍 **High Accuracy** on BUSI & Distal Myopathy datasets  
- 🧠 **Attention-Aware Fusion** for improved interpretability
- 📊 **Functionally Grounded Metrics**: coherence score, incremental deletion

---

🌟 Inspiration
This work builds upon and extends ideas from:

RadFormer – combining transformers with radiology workflows

GitHub: https://github.com/sbasu276/RadFormer

Publication: Sharma et al., “RadFormer: Transformer-based Radiology Report Generation” (ScienceDirect)
https://www.sciencedirect.com/science/article/pii/S1361841522003048

Attention-Gated Networks – integrating attention gates into CNNs for medical imaging

GitHub: https://github.com/ozan-oktay/Attention-Gated-Networks

Paper: Oktay et al., “Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images”

