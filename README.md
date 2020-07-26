## Trajectory prediction challenge for SAIC 2020

### Introduction

We realize trajectory prediction by combining the method based on deep learning and physical model.Based on deep learning method,we have designed a simplified version prediction framework based on LSTM-encoder-decoder framework and a complete version prediction framework considering interaction model and multi-modal trajectory prediction.Considering the form of the match dataset,we use Frenet coordinate system in order to make full use of the lane centerline information.In training process,we pretrain network by Argoverse dataset and fine-tune network using SAIC dataset(match dataset).

### usage

1. SAIC_data_visual.py script is used to visualize the trajectory in the scene(each csv file).
2. SAIC_data_process.py script is used to preprocess the original match dataset,all the samples are divided into training dataset and validation dataset in a ratio of 8:2 for deep learning network training.
3. SAIC_model.py.We have implemented a simplified version prediction framework based on LSTM-encoder-decoder framework and a complete version prediction framework considering interaction model and multi-modal trajectory prediction.
4. SAIC_utils.py.We implement the transformation between the world absolute coordinate systerm and Frenet relative coordinate systerm.We realize different evaluation metrics and loss functions respectively for Argoverse dataset pretraining and SAIC dataset fine-tuning.
5. argoverse_pretrain.py script is used to pretrain deep learning network by Argoverse dataset.
6. SAIC_train.py script is used to fine-tune deep learning network based on Argoverse dataset pretraining by SAIC dataset.
7. SAIC_val.py script is used to  evaluate the prediction performance of the deep learning network over the validation dataset.
8. SAIC_test_process.py script is used to preprocess the original match dataset for testing(submit).
9. SAIC_test.py script is used to generate prediction trajectory for test dataset by deep learning method.
10. dump_test_csv.py is used to generate prediction trajectory for remaining test dataset by physical model and generate final csv files for submission.





