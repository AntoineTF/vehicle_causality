# UniTraj: Causal Analysis Framework for Trajectory Prediction

This folder extends the **Unitraj** framework to support **causal analysis** in trajectory prediction models. Our modifications include causal regularization techniques and counterfactual generation to evaluate the causal influence of individual agents in driving scenarios.

## Code Structure

```
unitraj/
├── configs/
│   ├── config.yaml                  # General configurations
│   ├── method/
│   │   ├── autobot.yaml             # AutoBot-specific configurations
│   │   ├── MTR.yaml
│   │   └── wayformer.yaml
│
├── datasets/
│   ├── base_dataset.py              # Base dataset loader with counterfactual support
│   ├── autobot_dataset.py
│   ├── wayformer_dataset.py
│   └── MTR_dataset.py
│
├── models/
│   ├── autobot/                     # Main folder for the AutoBot model
│   ├── mtr/
│   ├── wayformer/
│   └── base_model/                  # Modified for counterfactual handling and embeddings
│
├── utils/
└── train.py                         # Training script

```

## Modifications and Rationale

### **1. Base Model Adjustments (`base_model.py`):**

- **Training Step Changes:** The `training_step` method now processes **counterfactual embeddings** alongside factual data.
- **Counterfactual Handling:** The `return_embeddings` flag allows flexible handling of both data types.

### **2. Counterfactual Handling (`autobot.py`):**

- **Contrastive Projector:** Added a contrastive projector for generating embeddings suitable for causal analysis.
- **Forward Pass Changes (`_forward` method):** Processes both factual and counterfactual embeddings.
- The code overall is now modified to be **counterfactual friendly**.

### **3. Loss Functions:**

Two causal regularizers were introduced:

- **Contrastive Loss (`calc_contrastive_loss`)**: Encourages factual embeddings to be closer to non-causal counterfactuals and farther from causal ones.
- **Ranking Loss (`calc_ranking_loss`)**: Enforces a margin between factual and causal counterfactuals for better separation.


## Causal Metrics: ACE Calculation

The **Average Causal Effect (ACE)** metrics were introduced to quantify the causal relationships between agents.


### **1. Non-Causal Average Causal Effect (NC-ACE)**

Measures differences between factual and non-causal counterfactuals:

\[
NC\text{-}ACE = \frac{1}{|NC|}\sum_{i \in NC} \|f_i - f_{actual}\|_2
\]


### **2. Direct Causal Average Causal Effect (DC-ACE)**

Calculates differences between factual and direct causal counterfactuals:

\[
DC\text{-}ACE = \frac{1}{|DC|}\sum_{i \in DC} \|f_i - f_{actual}\|_2
\]


### **3. Overall ACE**

Combines both non-causal and direct causal effects:

\[
ACE = \frac{1}{|NC| + |DC|}\sum_{i \in (NC + DC)} \|f_i - f_{actual}\|_2
\]


### Key Terms:
- **Factual Prediction:** \( f_{actual} \) represents the standard prediction.  
- **Counterfactual Prediction:** \( f_i \) represents a counterfactual prediction under modified conditions.

## Citation

If you use this repository in your research, please cite **Unitraj**:

```
@article{feng2024unitraj,
  title={UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction},
  author={Feng, Lan and Bahari, Mohammadhossein and Amor, Kaouther Messaoud Ben and Zablocki, {\'E}loi and Cord, Matthieu and Alahi, Alexandre},
  journal={arXiv preprint arXiv:2403.15098},
  year={2024}
}
```
