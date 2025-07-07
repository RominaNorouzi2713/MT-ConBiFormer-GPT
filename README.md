# MT-ConBiFormer-GPT

![Encoder and Decoder Architecture](Encoder%20&%20Decoder%20Denovo%20recovered%20(5).png)


## Table of Contents
1. [Description](#description)
2. [Requirements](#requirements)
3. [Usage](#usage)


### Description<a name="description"></a>

 MT-ConBiFormer-GPT, an innovative generative framework specifically designed for low-data, multi-target molecular generation,
 with a focus on the PI3K–AKT–mTOR signaling pathway relevant to cancer. This model, based on a variational autoencoder (VAE)
 with a BiFormer-based encoder and a pre-trained SMILES-GPT decoder, uniquely generates molecules for more than two biological targets simultaneously,
 a first in the field.Its efficacy stems from a three-phase, data-efficient training strategy: unsupervised pretraining, supervised contrastive learning to organize the latent space,
 and curriculum-based fine-tuning that incrementally adapts the model from generating dual-target to triplet-target inhibitors for PIK3CA, AKT1, and MTOR.
 Evaluations confirm MT-ConBiFormer-GPT's superior performance in generating valid, unique, and diverse molecules, even demonstrating state-of-the-art generalization in an omics-driven drug design task. 
 This integration of advanced deep learning into a data-efficient workflow positions MT-ConBiFormer-GPT as a robust solution
 for accelerating the identification of novel multi-target therapeutics.

### Requirements<a name="requirements"></a>

To set up the necessary environment to run this project, first clone the repository and then install the required packages from the `requirements.txt` file.

```bash
git clone https://github.com/RominaNorouzi2713/MT-ConBiFormer-GPT.git](https://github.com/RominaNorouzi2713/MT-ConBiFormer-GPT.git
cd MT-ConBiFormer-GPT
pip install -r requirements.txt
