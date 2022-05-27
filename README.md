# Do Regularizers improve Reproducibility in Neural Networks?

In this work, we empirically find out if common regularizers help to improve
*test prediction consistency* across several runs. i.e: **prediction difference**

This is in contrast to the common convention that benchmarks regularizers on 
test prediction consistency with respect to the ground truth predictions. i.e: test accuracy

Intutitively, regularizers *stabilize* the learning process. 

So if we eliminate most sources of variability, 
by controlling the seeds used during training, 
such as: those used in the random initialization and sampling process
we would expect to obtain consistent test predictions, no matter how many times, we train the network.

However our findings show that this may not be so. 
We find that despite, seed control, the model (representations) found by the neural network learning process varies each time it is trained.

Interestingly, even though the test accuracies across training runs seem close with respect to some decimal places. The prediction difference most always varies.

Consequently, we introduce a newer metric: *Effective Test Accuracy* as a better measure of performance compared to *Test Accuracy*.

*Effective Test Accuracy*  is the test accuracy measure of a neural net model subject to its prediction differences across an integer number of training runs.

We hope, our findings can add to the adoption of *Effective Test Accuracy* by the research community when reporting neural network modeling results.

Use conda create -n <name of env> --file package-list.txt to create a conda environment with installed packages necessary to run the code
Use conda activate <name of env> to activate the environment


# Technical Notes

[-] core python modules 
```
numpy==1.22.3
dash==2.3.1
scipy==1.8.0
torch==1.11.0+cu113
```
