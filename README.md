# Multimodal-Attention-Aware-Interpretability
A deep learning framework for diagnosing distal myopathy with enhanced explainability via attention-gated multimodal fusion. Our architecture integrates global and local feature extractors and generates saliency maps whose interpretability is evaluated both quantitatively and by expert radiologists.

🔍 Overview
Distal myopathies are genetically heterogeneous skeletal muscle disorders characterized by specific myofiber-level abnormalities. Accurate, explainable AI (XAI) tools can support clinicians by not only predicting disease presence but also highlighting relevant regions of interest.

Key contributions:

Attention-Gated Fusion: Combines a global ResNet50 backbone with a local BagNet33 via adaptive attention gates.

Dual Interpretability Evaluation:

Functionally Grounded: coherence scoring against reference masks; MoRF/LeRF incremental-deletion analysis.

Application-Grounded: radiologist survey and thematic analysis of saliency-map usefulness and trust.

Clinical Validation: Involves seven radiologists (1–9 years’ experience) to assess map specificity, accuracy, and perceived trust.
