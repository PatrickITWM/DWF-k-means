# DWF-k-means

## Reference
[TODO] The code was used in [cite]

## Installation
Python 3.11.4 is required. Install all requirements in the requirements.txt file.

## How to use the code
### The flkm folder
The flkm folder can be treated as a python package.
We implemented EWF k-means and DWF k-means in the class FLKMeans, which can be used similarly to the scikit-learn 
k-means class. We also implemented k-FED in the class KFed with a similar interface and a function to compute the 
classical k-means score. In the utils, a method is provided to distribute the data with different distributions to 
the clients.
### The experiment files
To run the experiments, run the corresponding experiment_*.py file. 
Note that the code runs only on CPU, but the computation is parallelized.
Each trained model is stored in the folder savepoints using joblibs Memory function, 
which will be created if not existent, such that it will not be recomputed when run again.
If one needs to reset the savepoints, just delete the folder savepoints or the particular subfolder.
The results of the evaluated metrics for all runs are stored in csv files in the folder experiments.

To recreate the images used in the paper, run the evaluate.ipynb jupyter notebook.