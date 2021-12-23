# OutcomePrediction

## Installation
First commit of the code for the UCL-Magnae Greacia led collaboration on lung Outcome Prediction
In this code, we use conda and pytorch to uniformize effort.
To set up your Python environment:
- Install `conda` or `miniconda` for your operating system.

For Linux:
- Create a Conda environment from the `environment.yml` file in the repository root, and activate it:
```shell script
conda env create --file environment.yml
conda activate UCL_DL
```
For Windows:
- Create a Conda environment from the `environment_win.yml` file in the repository root, and activate it:
```shell script
conda env create --file environment_win.yml
conda activate UCL_DL
```

To then set up your file as a module, please use
```shell script
conda-develop /path/to/OutcomePrediction/
```

Data comments:
- Patient 0617-444138 is very strange (extrememly short CT). The max dose crop is not working, it is better removing it from the mastersheet file.
