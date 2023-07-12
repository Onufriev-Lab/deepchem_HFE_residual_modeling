
This library contains a series of files showing how experiments were prepared, run and evaluated

1. Dictonary Generation
  - appends thr results of various physics based models into the FreeSolv dataset
2. Partitioning
  - demonstrates how the daataset was partitioned in such a way that each partition represents an even distribution of the whole dataset
3. Error Prediction
  - a simple experiment predicting the residuals of a physics model and correcting the values
4. K-fold Cross Validation
  - using k-fold cross validation to run multiple experiments and gain a more general understanding of the models preformance on the dataset
5. Hyperparameter Exploration
  - running many cross validation experiments to
6. Hyperparameter Evaluation
  - train and store models on final hyperparameters


Folders
  - Graphs
  - Experiments
  - Models
  - Dicts

MSC
  - consol.pickle - consolidated physics based experiments and FreeSolv dictionary
  - utils.py      - msc utility functions
  - graphing.py   - graphing utilities