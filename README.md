# Do Regularizers improve Reproducibility in Neural Networks?


Intuitively, we start with the hypothesis that regularizers induce some amount of stability in the learning process, the learning process determines the weights (representations) in the neural network, and the weights determine the predictions of the network.

Then if we were to attenuate most sources of variability during training, by controlling the random generator seeds used, such as: those used in the random initialization and sampling process; we would expect that compared to when the explicitly added regularizers were absent, we should obtain some amount of increased reproducibility on the network's predictions (less prediction difference) on the test set across training runs, irrespective of the number of times we were to train the network.

<!-- % by empirically finding out if commonly used regularizers help to improve test prediction consistency (or reduce prediction difference) across several training runs. -->

In this work, we attempt to prove the aforementioned intuitive expectation. We empirically find out if common regularizers help to improve
*test prediction consistency* across several runs. i.e: *Prediction Difference*. This is in contrast to the common convention that benchmarks regularizers on test prediction consistency with respect to the ground truth predictions. i.e: *0/1 Test Accuracy* or *0/1 Test Loss*


Our empirical findings show that such hypothesis can be significantly rejected. We find that, despite recommended seed control, the model (representations) found by the neural network learning process varies each time it is trained. Interestingly, even though the test accuracy across training runs seem close with respect to some decimal places. The test prediction difference most always varies. To capture the effect of such variability when training deep neural networks, we introduce a newer metric, *Effective Test Accuracy*, as a better measure of trusting test performance results compared to using only the popular *Test Accuracy* metric.

*Effective Test Accuracy*  is the test accuracy measure of a neural net subject to its prediction differences across a number (integer) of training runs. We hope, our findings can add to the adoption of *Effective Test Accuracy* by the research community when reporting neural network modeling results.



# Technical Notes

The following contains instruction that aid in the attempt to reproduce the python environment for this repo. Tested on Python Version: 3.10.

The **[-]** prefix is used here to indicate a command to be entered in a terminal (e.g: bash or powershell or cmd).

### Conda

- [x] To create a *conda* environment with the necessary dependencies needed run the code

[-] ```conda create -n <name of env> --file package-list.txt```

- [x] To activate the *conda* environment

[-] ```conda activate <name of env>``` 

### Pip

Also, see https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/


- [x] Create a *venv* in any suitable directory. For example: desired *venv* name could be *regrep*

[-] ``` python -m venv regrep ```

- [x] for Linux

  [-] ``` source regrep/bin/activate && alias python='regrep/bin/python3.x' ```

- [x]  for Windows PowerShell

  [-] ```.\regrep\Scripts\activate```  

- [x] Install Dependencies

[-] ```pip install --upgrade pip```

[-] ```pip install -r package-list.txt```


- [x] Run the experiments

[-] ```python ex_mlp.py```

- [x] Leave the *venv*

[-] ```deactivate```

- [x] Delete the *venv*

[-] ```rm -rf regrep```


### Core python modules 
```
numpy==1.22.3
dash==2.3.1
scipy==1.8.0
torch==1.11.0+cu113
```
