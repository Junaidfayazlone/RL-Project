#!/usr/bin/env bash

# data_input_dir="datasets/data_preprocessed/umls/"
data_input_dir="datasets/data_preprocessed/umls_inv/"
# vocab_dir="datasets/data_preprocessed/umls/vocab"
vocab_dir="datasets/data_preprocessed/umls_inv/vocab"
total_iterations=2000
path_length=2
hidden_size=50
embedding_size=50
batch_size=256
beta=0.05
Lambda=0.05
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
# base_output_dir="output/umls/"
# base_output_dir="train_results/umls/"
base_output_dir="train_results/umls_inv/"
load_model=0
# load_model=1
# model_load_dir="/home/sdhuliawala/logs/RL-Path-RNN/uuuu/8fe2_2_0.06_10_0.02/model/model.ckpt"
model_load_dir="/home/sgsharma/Documents/sem3/rl/project/MINERVA/train_results/umls/8ee3_2_0.05_100_0.05/model/model.ckpt"
nell_evaluation=0
