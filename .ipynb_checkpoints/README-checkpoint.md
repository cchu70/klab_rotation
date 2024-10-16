# klab_rotation

1. Classifier directly on pixels (SVM)
2. Copy architecture from Duncan (https://klab.tch.harvard.edu/publications/PDFs/gk7817.pdf) using backprop
3. Implement growth
4. Implement pruning

# Notes

2024-10-15: 
- Set up conda environment and git repo.
- Implemented simple SVM for MNIST classification (01_Basic-SVM-MNIST-Classification.ipynb)
    - Took a while to run grid search locally. Reduced the training set to only 1k data points.
- Implemented simple MLP using keras and parameters mentioned in the thesis document. I was unsure if he used ReLU for the hidden layer and softmax for the output. 