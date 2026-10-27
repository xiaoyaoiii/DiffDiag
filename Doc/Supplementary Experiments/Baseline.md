## 📏 Baselines

We compare **DiffDiag** with representative baselines from both traditional medical consultation methods and LLM-based consultation frameworks.

**EBAD**: An evidence-based dialogue system that learns symptom inquiry policies through reinforcement learning and selects symptoms by optimizing consultation rewards.

**MTDiag**: A multi-task automatic diagnosis framework that jointly models symptom inquiry and disease prediction, enabling the consultation agent to collect evidence while optimizing diagnostic performance.

**HAIformer**: A human-AI collaborative diagnosis framework that incorporates doctor-enhanced transformer representations to improve disease prediction from collected symptom evidence.

**Diaformer**: A generation-based automatic diagnosis model that formulates medical consultation as symptom sequence generation, using previously observed symptoms to predict subsequent symptoms and diseases.

**DxFormer**: A decoupled automatic diagnosis framework that separates symptom inquiry from disease prediction and leverages dense symptom representations for medical consultation.

**Direct Prompting (DP)**: Directly prompts large language models to conduct symptom inquiry and disease diagnosis without additional task-specific training.

**Uncertainty of Thoughts (UoT)**: Enhances LLM-based consultation by introducing uncertainty-aware planning, encouraging the model to seek additional information when diagnostic confidence is insufficient.

**Chain of Diagnosis (CoD)**: Guides LLMs to perform interpretable medical reasoning through chain-of-diagnosis prompting, producing symptom inquiries and diagnostic decisions in a structured reasoning process.

**Dual-Decision Optimization (DDO)**: A multi-agent LLM-based medical consultation framework that decouples symptom inquiry and disease diagnosis, using an external reinforcement learning policy agent for inquiry action selection.
