# Task 2: Computer vision. Sentinel-2 image matching

- In this directory, there is a file named `task2.ipynb`, which contains all the data preprocessing steps, algorithm building, and an example image matching.
- Additionally, there are two Python files:
  - `algorithm_pipeline.py`, which is a copy of the data preprocessing, algorithm building and evaluating code from `task2.ipynb`.
  - `model_inference.py`, which is a copy of the code used to demonstrate the model's inference. For this script to work in this directory, a file containing the model results named `results.npy` must be present. The model results are available [here](https://drive.google.com/file/d/1W3uUFX9LhG63i0zN6Ts6zC5OTRgCB5X_/view?usp=sharing).
- For this task, I used the LoFTR model, which I used on little crops of satelite snapshots.
- The necessary libraries are listed in the requirements.txt file located in the task-2 directory.
- A detailed discussion of the dataset and model can be found in `task2.ipynb` and the accompanying PDF report.