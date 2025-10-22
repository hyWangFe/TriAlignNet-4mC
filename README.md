# TriAlignNet-4mC 
### Title:Prediction of DNA N4-Methylcytosine Sites Based on a Position-Aligned Multi-Branch Fusion Network  
  
**TriAlignNet-4mC** is a deep learning framework for the accurate identification of DNA N4-methylcytosine (4mC) sites in the *Mus musculus* genome.  
The model integrates biochemical, contextual, and structural features through a position-aligned multi-branch neural architecture to capture local sequence patterns, long-range dependencies, and DNA structural topology.




## Environment:
python  3.9  
pytorch  2.5.1  
numpy  1.26.3  
pandas  2.3.1  
cuda 12.1

## File Description:
1.data:Data used in the model    
2.feature_extract:Three-branch feature encoding method and pre-training model  
3.Data_process.py:Handles dataset loading, preprocessing, and splitting, including positive/negative sample construction and normalization.  
4.model2.py:Defines the TriAlignNet-4mC architecture, including the three-branch feature extraction modules and the position-aligned fusion layer.  
5.train2.py:Main training script that loads data, initializes the model, runs the training loop, and saves the best-performing weights.  
6.test2.py:Evaluation script for testing the trained model on cross-validation or independent datasets.  
7.utils.py:Provides utility functions for model saving, metric calculation, logging, and visualization.  
