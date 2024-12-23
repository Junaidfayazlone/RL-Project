# MINERVA
Meandering In Networks of Entities to Reach Verisimilar Answers 

Code and models for the paper [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851)


## Requirements
To install the various python dependencies (including tensorflow)
```
pip install -r requirements.txt
```

## Training
Training MINERVA is easy!. The hyperparam configs for each experiments are in the [configs](https://github.com/shehzaadzd/MINERVA/tree/master/configs) directory. To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the `${dataset}.sh` is the name of the config file. For example, 
```
sh run.sh configs/countries_s3.sh
```

## Testing

We are also releasing pre-trained models so that you can directly use MINERVA for query answering. They are located in the  [saved_models](https://github.com/shehzaadzd/MINERVA/tree/master/saved_models) directory. To load the model, set the ```load_model``` to 1 in the config file (default value 0) and ```model_load_dir``` to point to the saved_model. For example in [configs/countries_s2.sh](https://github.com/shehzaadzd/MINERVA/blob/master/configs/countries_s2.sh), make
```
load_model=1
model_load_dir="saved_models/countries_s2/model.ckpt"
```
## Output
The code outputs the evaluation of MINERVA on the datasets provided. The metrics used for evaluation are Hits@{1,3,5,10,20} and MRR (which in the case of Countries is AUC-PR). Along with this, the code also outputs the answers MINERVA reached in a file.

## Code Structure

The structure of the code is as follows
```
Code
├── Model
│    ├── Trainer
│    ├── Agent
│    ├── Environment
│    └── Baseline
├── Data
│    ├── Grapher
│    ├── Batcher
│    └── Data Preprocessing scripts
│            ├── create_vocab
│            ├── create_graph
│            ├── Trainer
│            └── Baseline

```

## Data Format

To run MINERVA on a custom graph based dataset, you would need the graph and the queries as triples in the form of (e<sub>1</sub>,r, e<sub>2</sub>).
Where e<sub>1</sub>, and e<sub>2</sub> are _nodes_ connected by the _edge_ r.
The vocab can of the dataset can be created using the `create_vocab.py` file found in `data/data preprocessing scripts`. The vocab needs to be stores in the json format `{'entity/relation': ID}`.
The following shows the directory structure of the Kinship dataset.

```
kinship
    ├── graph.txt
    ├── train.txt
    ├── dev.txt
    ├── test.txt
    └── Vocab
            ├── entity_vocab.json
            └── relation_vocab.json
``` 
## Citation
If you use this code, please cite our paper
```
@inproceedings{minerva,
  title = {Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning},
  author = {Das, Rajarshi and Dhuliawala, Shehzaad and Zaheer, Manzil and Vilnis, Luke and Durugkar, Ishan and Krishnamurthy, Akshay and Smola, Alex and McCallum, Andrew},
  booktitle = {ICLR},
  year = 2018
}
```
