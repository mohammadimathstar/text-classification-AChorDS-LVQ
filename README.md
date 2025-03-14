# Text Classification Pipeline using AChorDS-LVQ and DVC

## Overview
This repository contains a **text classification pipeline** using **AChorDS-LVQ** (Adaptive Chord Distance-based Learning Vector Quantization). The pipeline is managed using **DVC (Data Version Control)** for better reproducibility and tracking of experiments.

## Repository Structure
```bash
├── data/               # Folder containing datasets (not included in repo)
├── src/                # Source code for the pipeline
├── models/             # Trained models storage
├── reports/            # Folder containing metrics and top words (as the model is explainable, it highlights the most important words)
├── dvc.yaml            # DVC pipeline stages
├── params.yaml         # Configuration parameters for the pipeline
├── requirements.txt    # Required Python packages
├── README.md           # This file
```

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### Install Dependencies
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Install & Configure DVC
DVC helps track data and models efficiently.
```bash
pip install dvc
```
If using a remote storage (like Google Drive, AWS S3, or Azure), configure it:
```bash
dvc remote add -d myremote <remote-storage-url>
```

## Running the Pipeline
### Step 1: Pull Data and Models
If data and models are tracked using **DVC**, pull them:
```bash
dvc pull
```

### Step 2: Run the Pipeline
Execute the pipeline stages as defined in `dvc.yaml`:
```bash
dvc repro
```
This will automatically run all stages in the correct order, such as:
- Data preprocessing
- Feature extraction
- Model training (AChorDS-LVQ)
- Model evaluation

### Step 3: View Parameters
Check and modify pipeline parameters in `params.yaml` as needed before running the pipeline.
```yaml
learning_rate_prototypes: 0.01
learning_rate_lambda: 0.00001
num_epochs: 50
num_classes: 2
```

## Tracking & Experimentation
To track different experiment runs:
```bash
dvc metrics show
```
To compare results of different experiment versions:
```bash
dvc metrics diff
```

## Updating the Pipeline
If you modify the pipeline or update data, track changes:
```bash
dvc add data/
git add .
git commit -m "Updated dataset"
dvc push
```

## Contributions
Feel free to fork this repository and submit pull requests with improvements!

## Reference:

When citing our work, please refer to the following article:

`
@article{mohammadi2024prototype,
  title={A prototype-based model for set classification},
  author={Mohammadi, Mohammad and Ghosh, Sreejita},
  journal={arXiv preprint arXiv:2408.13720},
  year={2024}
}
`

Please use this article as the primary reference for our methodology and approach. If you need further details or clarification, don't hesitate to reach out.


---
### Need Help?
If you have any issues, feel free to open an [issue](https://github.com/your-repo/issues) or reach out! 🚀

