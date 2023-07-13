# Enhancing Yes/No Question Answering with Weak Supervision via Extractive Question Answering (Source Code)

## Setup Instructions

The following steps outline how to set up the source code of the paper on a Linux Ubuntu machine. Windows users can also follow similar steps, although we have not tested it on Windows or other operating systems.

1. Install Python 3.8 or a later version.
2. Make sure you have the appropriate GPU drivers installed on your machine.
3. Use the pip package manager to install the necessary libraries specified in the requirements.txt file.
4. Obtain the BoolQ dev and test datasets. Place them in a folder named "dataset" and name the files as bool_dev_set.jsonl and boolq_test.jsonl, respectively.
5. Execute the retrieve_models.py script to download the BERT and RoBERTA models.

## Verify Functionality

As the training process can be time-consuming, it is advisable to ensure that all Python modules are functioning correctly. To test the setup, please follow the steps below:

1. Open the file utils/dataset_handler.py and remove the comment symbols (#) from lines 14, 16, and 18, making the necessary modifications to the code.
2. Open the file utils/testing.py and remove the comment symbols (#) from lines 19 and 44.

By doing so, you can test the setup using a smaller subset of examples instead of the entire BoolQ dataset.

## Repository Overview

This repository comprises the source code for four distinct models discussed in the accompanying paper. To train and test each model, you can execute the corresponding files located in the root of the project: run_bert_base.py, run_bert_method.py, run_roberta_base.py, and run_roberta_method.py. The utils folder contains Python modules utilized for backend purposes.

1. bert_base_model.py: The Bert Base model class.
2. bert_model_method.py: The model class with our extension.
3. roberta_model_base.py and roberta_model_method.py: Similar to the Bert models.
4. dataset_handler.py: Responsible for handling the dataset, including examples and labels.
5. train.py, validate.py, and testing.py: Functions required for training, validating, and testing the models.

The dataset folder contains the data used for training, validating, and testing the models.

The models folder stores the utilized learning models.

The results folder contains the final outcomes obtained after training and testing a learning model.

The test_results folder includes the results obtained from complete experimentation.

## Modifying the Source Code

If you need to adjust the batch size for training the learning model, you will have to make the changes manually in the corresponding source files located in the root folder.






