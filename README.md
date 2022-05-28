# Do Regularizers improve Reproducibility in Neural Networks?

In this work, we empirically find out if common regularizers help to improve
*test prediction consistency* across several runs. i.e: *Prediction Difference*

This is in contrast to the common convention that benchmarks regularizers on 
test prediction consistency with respect to the ground truth predictions. i.e: *0/1 Test Accuracy* or *0/1 Test Loss*

Intutitively, regularizers *stabilize* the learning process. 

So if we eliminate most sources of variability, 
by controlling the seeds used during training, 
such as: those used in the random initialization and sampling process
we would expect to obtain consistent test predictions, no matter how many times, we train the network.

However our findings show that this may not be so. 
We find that despite, seed control, the model (representations) found by the neural network learning process varies each time it is trained.

Interestingly, even though the test accuracies across training runs seem close with respect to some decimal places. The prediction difference most always varies.

Consequently, we introduce a newer metric: *Effective (0/1) Test Accuracy* as a better measure of *trusting* the performance compared to just *Test Accuracy*.

*Effective Test Accuracy*  is the test accuracy measure of a neural net model subject to its prediction differences across an number (integer) of training runs.

We hope, our findings can add to the adoption of *Effective Test Accuracy* by the research community when reporting neural network modeling results.



# Technical Notes

### Conda
To create a *conda* environment with the necessary dependencies needed run the code

[-] ```conda create -n <name of env> --file package-list.txt```

To activate the *conda* environment

[-] ```conda activate <name of env>``` 

### Pip

Todo...

### Core python modules 
```
numpy==1.22.3
dash==2.3.1
scipy==1.8.0
torch==1.11.0+cu113
```
