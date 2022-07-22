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
Then, run `main.py` with command line arguments if desired 
```
python code/main.py --epochs <EPOCHS> --dataset <DATASET> --model <MODEL> 
    --batch_size <BATCH_SIZE> --eval_every <EVAL_EVERY>
```

