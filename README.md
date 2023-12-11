# Towards Effective and General Graph Unlearning via Mutual Evolution

**Requirements**

Hardware environment: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz, NVIDIA GeForce RTX 3090 with 24GB.

Software environment: Ubuntu 20.04.5, Python 3.8.10, Pytorch 1.13.0, and CUDA 11.7.0.
  1. Please refer to PyTorch and PyG to install the environments;
  
  2. Run 'pip install -r requirements.txt' to download required packages;

**Training**
To train model(s) in the paper
  1. Please unzip xxx.zip to the current file directory location
  2. Please refer to the configs folds to modify the hyperparameters

     config.py - Setting the path for data loading and saving.

     parameter_parser.py - Parameter settings for modeling and training.

  3. Open main.py to start unlearning

     We provide Cora dataset as example.

     Meanwhile, you can personalize your settings (lib_dataset/lib_gnn_model/exp).

     Run this command:

```
python main.py
```
