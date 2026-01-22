# Dual-Stream-Attention-Aware-Interpretability
_A deep learning framework for interpretable diagnosis of distal myopathy via dual-stream attention-aware fusion._

---

## 📝 Overview

**Distal myopathies** are genetically heterogeneous muscle disorders characterized by specific myofiber alterations.  
This repository implements a **Dual-Stream Attention-Aware Fusion** model that:  
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

## 🌟 Inspiration

This work draws on the following projects and publications:

1. **RadFormer** – Transformers with global–local attention for interpretable and accurate cancer detection  
   - GitHub: [sbasu276/RadFormer](https://github.com/sbasu276/RadFormer)  
   - Paper: Basu _et al._ (2023). RadFormer: Transformers with global–local attention for interpretable and accurate Gallbladder Cancer detection. *Medical Image Analysis*, 83, 102676.  

2. **Attention Gated Networks** – Attention gates for highlighting salient regions in CNN-based medical imaging  
   - GitHub: [ozan-oktay/Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks)  
   - Paper: Schlemper _et al._ (2019). Attention gated networks: Learning to leverage salient regions in medical images. *Medical Image Analysis*, 53, 197–207.

