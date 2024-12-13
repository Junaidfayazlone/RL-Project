#!/usr/bin/env bash
# source $1
# export PYTHONPATH="."
#!/usr/bin/env bash

#!/usr/bin/env bash

#!/usr/bin/env bash

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



echo "Base output directory: $base_output_dir"
cmd="python code/model/trainer.py --base_output_dir $base_output_dir --path_length $path_length --hidden_size $hidden_size --embedding_size $embedding_size \
    --batch_size $batch_size --beta $beta --Lambda $Lambda --use_entity_embeddings $use_entity_embeddings \
    --train_entity_embeddings $train_entity_embeddings --train_relation_embeddings $train_relation_embeddings \
    --data_input_dir $data_input_dir --vocab_dir $vocab_dir --model_load_dir $model_load_dir --load_model $load_model --total_iterations $total_iterations --nell_evaluation $nell_evaluation"



echo "Executing $cmd"

CUDA_VISIBLE_DEVICES=$gpu_id $cmd
