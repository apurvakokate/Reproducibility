# Do Regularizers improve Reproducibility in Neural Networks?

In this work, we empirically find out if common regularizers help to improve
test prediction consistency across several runs. i.e: prediction difference

This is in contrast to the common convention that benchmarks regularizers on 
test prediction consistency with respect to the ground truth predictions. i.e: test accuracy

Intutitively, regularizers stabilize the learning process, so we would expect that,
if we eliminate most sources of variability by controlling the seeds used during training, 
such as those used in the random initialization and sampling process
we would obtain consistent test predictions, no matter how many times, we train the network.

However our findings show that this is not so. 
We find that despite, seed control, the model found by the network varies each time it is trained.
Interestingly, even though test accuracy is close as much as possible. The prediction difference most always varies

We hope, our findings can add to the adoption of prediction difference tests when reporting neural net results.

