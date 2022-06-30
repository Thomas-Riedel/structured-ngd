# Structured Natural Gradient Descent

Master Thesis "Structured Natural Gradient Descent for Bayesian Deep Learning" by Thomas Riedel; Supervised by Yuesong Shen; Examined by Prof. Dr. Daniel Cremers

## Compiling the latex document
Clone the repository and change into it and the `thesis` folder
```
git clone <REPO_URL>
cd structured-ngd
```
Then, run
```
pdflatex MA_ThomasRiedel.tex
biber MA_ThomasRiedel
pdflatex MA_ThomasRiedel.tex
pdflatex MA_ThomasRiedel.tex
```
and open the generated `.pdf` file.

## Running the code
Clone the repository and cd into it as described above.
Then, run `main.py` with command line arguments if desired 
```
python code/main.py --epochs <EPOCHS> --dataset <DATASET> --model <MODEL> 
    --batch_size <BATCH_SIZE> --eval_every <EVAL_EVERY>
```

