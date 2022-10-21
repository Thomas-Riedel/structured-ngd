# Structured Natural Gradient Descent

Master Thesis "Structured Natural Gradient Descent for Bayesian Deep Learning" by Thomas Riedel; Supervised by Yuesong Shen; Examined by Prof. Dr. Daniel Cremers

## Compiling the latex document
Clone the repository and change into it and the project folder
```
git clone https://gitlab.lrz.de/00000000014965FA/structured-ngd.git
cd structured-ngd
```
Then, run
```
pdflatex thesis/MA_ThomasRiedel.tex
biber thesis/MA_ThomasRiedel
pdflatex thesis/MA_ThomasRiedel.tex
pdflatex thesis/MA_ThomasRiedel.tex
```
and open the generated `MA_ThomasRiedel.pdf` file in the `thesis` folder.

## Running the code
Clone the repository and cd into it as described above.
Command line arguments can be specified if desired:
```
usage: main.py [-h] [-o OPTIMIZERS] [-e EPOCHS] [-d DATASET] [-m MODEL]
               [--batch_size BATCH_SIZE] [--lr LR] [--k K]
               [--mc_samples MC_SAMPLES] [--structure STRUCTURE]
               [--eval_every EVAL_EVERY] [--momentum_grad MOMENTUM_GRAD]
               [--momentum_prec MOMENTUM_PREC]
               [--prior_precision PRIOR_PRECISION] [--damping DAMPING]
               [--gamma GAMMA] [-s DATA_SPLIT] [--n_bins N_BINS]
Run noisy optimizers with parameters.
optional arguments:
  -h, --help            show this help message and exit
  -o OPTIMIZERS, --optimizers OPTIMIZERS
                        Optimizers, one of Adam, StructuredNGD (capitalization
                        matters!, default: StructuredNGD)
  -e EPOCHS, --epochs EPOCHS
                        Number of epochs to train models on data (default: 1)
  -d DATASET, --dataset DATASET
                        Dataset for training, one of CIFAR10, MNIST,
                        FashionMNIST (default: CIFAR10)
  -m MODEL, --model MODEL
                        ResNet model (default: resnet20)
  --batch_size BATCH_SIZE
                        Batch size for data loaders (default: 64)
  --lr LR               Learning rate (default: 0.1)
  --k K                 Rank parameter for StructuredNGD (default: 0)
  --mc_samples MC_SAMPLES
                        Number of MC samples (default: 1)
  --structure STRUCTURE
                        Covariance structure (default: rank_cov)
  --eval_every EVAL_EVERY
                        Frequency of summary statistics printing during
                        training (default: 100)
  --momentum_grad MOMENTUM_GRAD
                        First moment strength (default: 0.9)
  --momentum_prec MOMENTUM_PREC
                        Second moment strength (default: 0.999)
  --prior_precision PRIOR_PRECISION
                        Spherical prior precision (default: 0.4)
  --damping DAMPING     Damping strength for matrix inversion (default: 0.01)
  --gamma GAMMA         Regularization parameter in ELBO (default: 1.0)
  -s DATA_SPLIT, --data_split DATA_SPLIT
                        Data split for training and validation set (default:
                        0.8)
  --n_bins N_BINS       Number of bins for reliability diagrams (default: 20)
```

The script can be run using these arguments as follows:
```
python code/main.py <ARGUMENTS>
```

