# CI/CD for machine learning models

So far, we have been writing pipelines and tests for the model _code_ - but what about the (trained) models? In this part, we look at how CI/CD can be extended to the aspects of machine learning that are beyond coding.
For ML models, CI/CD concepts can be utilized to streamline the process of model training (and retraining), data and model validation and model deployment:

- Data integrity validation: Assure that the data does not contain errors or suffer from problems introduced during the data gathering process (e.g. bugs in the data processing pipeline or changes in the data source).
- Dataset comparisons: Assure that the dataset is not drifting / spot data drift early.
- Model training: Automate the model's training procedure on new (validated) training data.
- Model validation: Evaluate the model's performance, calibrating model predictions, and identifying weak segments.
- Model deployment

Here, we focus on CI for model training and validation. Data and deployment will be discussed in the upcoming labs.

## How to test a model

If you have ever taken a course on machine learning, you are likely familiar with the concept of a _test_ set. If not: A test set is a portion of the dataset that is held out from the training process and used to evaluate the _performance_ of a trained model. The test set serves as an independent dataset to assess how well the model generalizes to new, unseen data. Typically, this looks something like the snippet below.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)
```

You might be inclined to say that you do not need a pipeline for this - and you'd be right!
But this is really only the beginning, especially as the models become more complex, more aspects become relevant:

- Does your model perform equally well on all subsets (segments) of the data?
- Are there unused features?
- Is the model inference time increasing?
- ... and many more.

Luckily, there are tools that give you a helping hand when it comes to creating test suites for your machine learning models. Here, we will work with [deepchecks](https://docs.deepchecks.com/stable/getting-started/welcome.html). In their own words,

> Deepchecks is a holistic open-source solution for all of your AI & ML validation needs, enabling you to thoroughly test your data and models from research to production.

### Testing models with Deepchecks

Deepchecks implements _test suites_ for your models. A test suite consists of one or more _checks_. 

---

The final two sections introduce you to two tools that enhance CI/CD pipelines for machine learning systems.

## Continuous machine learning with CML

[Continuous Machine Learning (CML)](https://cml.dev) is a tool that extends common CI/CD solutions (GitHub Actions, GitLab CI/CD, and Bitbucket Pipelines) to machine learning. The idea behind CML is that you use the CI/CD pipeline for training and evaluating your models in the same way that you use them to build and test your code. The `deepchecks` pipeline from above could be enhanced as follows:

TODO: Change this!

```yaml
name: CML
on: [push]
jobs:
  train-and-report:
    runs-on: ubuntu-latest
    # optionally use a convenient Ubuntu LTS + DVC + CML container
    # container: docker://ghcr.io/iterative/cml:0-dvc2-base1
    steps:
      # may need to setup Node.js & Python3 on e.g. self-hosted
      # - uses: actions/setup-node@v2
      #   with:
      #     node-version: '16'
      # - uses: actions/setup-python@v4
      #   with:
      #     python-version: '3.x'
      - uses: iterative/setup-cml@v1
      - uses: actions/checkout@v3
      - name: Train model
        run: |
          # Your ML workflow goes here
          pip install -r requirements.txt
          python train.py
      - name: Create CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Post CML report as a comment in GitHub
          cat metrics.txt >> report.md
          echo '![](./plot.png "Confusion Matrix")' >> report.md
          cml comment create report.md
```

This workflow does the following:

1. Set up CML in the GitHub actions workflow.
2. Checkout the code.
3. Install dependencies and execute `train.py`.
4. Report metrics and plots created during training to GitHub.

The only part where CML really comes into play here is the last step: CML makes it easy to generate reports in response to changes. `cml comment create` posts a markdown report as a comment on a commit, pull/merge request, or issue. The result looks like the image below.

![CML report example](imgs/cml_report_example.png)

[CML also comes with utilities to launch dedicated runners](https://cml.dev/doc/ref/runner) for the workflow - this is especially useful when using CML for neural networks as GitHub's runners do not come with GPUs.
However, the tool introduced next sections specializes in this discipline!

## SkyPilot: Abstract away the infrastructure

In their own words,
> SkyPilot is a framework for running LLMs, AI, and batch jobs on any cloud, offering maximum cost savings, highest GPU availability, and managed execution.

SkyPilot abstracts away the cloud infrastructure (deploying VMs or clusters, allocating storage, etc) and provides a unified interface to launch jobs on this infrastructure. A SkyPilot _task_ specifies: resource requirements, data to be synced, setup commands, and the task commands. Once written, the task can be launched on any available cloud.

Using SkyPilot, the training job in our pipeline from above would looks something like the yaml snippet below:

TODO: Change this!

```yaml
# Declare which resources (e.g. GPUs you want)
resources:
  accelerators: V100:1  # 1x NVIDIA V100 GPU

# Say how many nodes you need.
num_nodes: 1  # Number of VMs to launch

# Working directory (optional) containing the project codebase.
# Its contents are synced to ~/sky_workdir/ on the cluster.
workdir: ~/torch_examples

# Commands to be run before executing the job.
# Typical use: pip install -r requirements.txt, git clone, etc.
setup: |
  pip install "torch<2.2" torchvision --index-url https://download.pytorch.org/whl/cu121

# Commands to run as a job.
# Typical use: launch the main program.
run: |
  cd mnist
  python main.py --epochs 1
```

Then, the workflow becomes:

```yaml
name: CML
on: [push]
jobs:
  train-and-report:
    runs-on: ubuntu-latest
    # optionally use a convenient Ubuntu LTS + DVC + CML container
    # container: docker://ghcr.io/iterative/cml:0-dvc2-base1
    steps:
      # may need to setup Node.js & Python3 on e.g. self-hosted
      # - uses: actions/setup-node@v2
      #   with:
      #     node-version: '16'
      # - uses: actions/setup-python@v4
      #   with:
      #     python-version: '3.x'
      - uses: iterative/setup-cml@v1
      - uses: actions/checkout@v3
      - name: Train model
        run: sky launch task.yaml
      - name: Create CML report
        env:
          REPO_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Post CML report as a comment in GitHub
          cat metrics.txt >> report.md
          echo '![](./plot.png "Confusion Matrix")' >> report.md
          cml comment create report.md
```
