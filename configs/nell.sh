#!/usr/bin/env bash

# data_input_dir="datasets/data_preprocessed/nell/"
data_input_dir="datasets/data_preprocessed/nell-995/"
# vocab_dir="datasets/data_preprocessed/nell/vocab"
vocab_dir="datasets/data_preprocessed/nell-995/vocab"
total_iterations=3000
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.02
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
# base_output_dir="output/nell/worksfor"
# base_output_dir="train_results/nell/worksfor"
base_output_dir="train_results/nell-995/worksfor"
# load_model=0
load_model=1
model_load_dir="/home/sgsharma/Documents/sem3/rl/project/MINERVA/train_results/nell-995/worksfor/2ada_3_0.05_100_0.02/model/model.ckpt"
# model_load_dir="output/nell/worksfor/de5b_3_0.05_100_0.02/model/model.ckpt"
nell_evaluation=0
