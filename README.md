# Coverage-Guided Metamorphic Testing for Pre-trained Language Models

> Final Project for Program Analysis (2020 Fall)

This repo contains a toolkit and data for the experiments described in our paper of "Coverage-Guided Metamorphic Testing for Pre-trained Language Models".

# Project Structures

Overall, our project is organized in the following structure.

- `SST-2` contains the raw data of the Stanford Sentiment Treebank v2 (SST2) dataset
- `filter_data` contains our manually filtered SST2 data for effective testing
- `metamorphicTesting` contains code for our transformations and metamorphic relations
- `tmp` is the output directory to host result files
- `main.py` is the entry program of our tool


# Install Dependency 

The tool is tested in three major platforms but only in python3.7. We recommend install it in a clean python environment.

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

# Configuration

In the entry program `main.py`, an entry function `draw_comparison(nsample=10, coverage_type=0)` is executed.
The default parameters here mean 10 random runs with the coverage metric indexed as follows.
- 0-DeepXplore
- 1-DeepGauge
- 2-NearestNeighbor

By default, the program runs in CPU mode, if you have access to GPU resource, enable GPU computation by setting `cuda=True` in `main.py:L241` and set the environment variable `CUDA_VISIBLE_DEVICES`.

# Run Experiment

Execute the entry program `main.py`, the program will start running the experiments with output files saved to the directory `tmp`.
Each experiment will produce three output files
- A csv file record all the coverage changes during the process
- A txt file record all failures trigered by the generated tests
- A png file illustrate coverage under each of three neuron coverage metrics changing with the number of generated tests

At the first time, the program will automatically download the DistilBERT model used in our tool.
Running the program in CPU could eat out lot of computation resource and take much longer (1min for one experiment unit) than the GPU mode.

# Experimental Results

The experimental results used in our paper are also included in the `saved_results` directory.