#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/kinship/"
vocab_dir="datasets/data_preprocessed/kinship/vocab"
total_iterations=2000
path_length=2
hidden_size=50
embedding_size=50
batch_size=512
beta=0.05
Lambda=0.05
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/kinship/"
load_model=0
model_load_dir="null"
nell_evaluation=0
