# B²F-Net: A Bidirectional and Bimodal Fusion Network for Weakly Supervised Video Violence Detection

Official implementation of our paper:

> **B²F-Net: A Bidirectional and Bimodal Fusion Network for Weakly Supervised Video Violence Detection**  
> Ju  et al.  
> Submitted to *Knowledge-Based Systems (KBS)*, 2025.

---

## Overview

This repository provides the implementation of **B²F-Net**, a weakly supervised video violence detection model.  
The network integrates **Temporal Convolutional Networks (TCN)** and **Bidirectional Gated Recurrent Units (BiGRU)**  
to model both short- and long-term temporal dependencies.

A **two-stage multimodal fusion strategy** is designed to combine raw and processed audio-visual features,  
enhancing temporal understanding while preserving complementary cross-modal information.

---

###  Training
If you want to train the model, please run:
```bash
nohup python main.py &
```

###  Inference
If you want to test the trained model, please run:
```bash
nohup python infer.py &
```

### Dataset
The XD-Violence dataset used in this work can be publicly accessed from
the paper by Wu et al., "Not Only Look, but Also Listen: Learning Multimodal Violence Detection under Weak Supervision."
