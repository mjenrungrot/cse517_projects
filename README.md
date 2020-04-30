## Reproducing EMNLP 2019 _Integrating Text and Image: Determining Multimodal Document Intent in Instagram Posts_

See our writeup ![here](writeup.pdf).

This project is to reproduce a paper entitled _Integrating Text and Image: Determining Multimodal Document Intent in Instagram Posts_ from EMNLP 2019. The project is a part of final project for CSE 517 Natural Language Processing (Winter 2020), offered at the University of Washington, Seattle.

## Installation

The only prerequisite is the installation of Anaconda.

1. `git clone --recurse-submodules https://github.com/mjenrungrot/cse517_projects` 
2. `conda env create -f environment.yml`
3. `conda activate cse517_projects`

## Optional: Run Tensorboard

In order to visualize the results across multiple splits, we recommend that you use tensorboard. To run a tensorboard, you can simply execute the following command. 
```
tensorboard --logdir ./logs --host 0.0.0.0
```
Then, you can access the Tensorboard visualization site at `http://[host]:[port_number]`.

## Get the reproduced results

Our implementation supports training on both CPU and GPU. Here's the sample command for training the network
on a single split for intent classification task and using both image and caption as inputs.

__CPU__ 
```
python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --classification intent \
    --tensorboard
```

__GPU__
```
python main.py --image_text \
    ./documentIntent_emnlp19/splits/train_split_0.json \
    ./documentIntent_emnlp19/splits/val_split_0.json \
    ./documentIntent_emnlp19/labels/intent_labels.json \
    ./documentIntent_emnlp19/labels/semiotic_labels.json \
    ./documentIntent_emnlp19/labels/contextual_labels.json \
    --device cuda:0 \
    --classification intent \
    --tensorboard
```

Here's the detailed description of parameters you can choose to run our implementation.
```
usage: main.py [-h]
               (--image_only | --image_text | --text_only | --elmo_image_text | --elmo_text_only)
               --classification CLASSIFICATION
               [--name NAME] [--epochs EPOCHS] [--lr LR]
               [--batch_size BATCH_SIZE] [--shuffle]
               [--num_workers NUM_WORKERS]
               [--device DEVICE]
               [--lr_scheduler_gamma LR_SCHEDULER_GAMMA]
               [--lr_scheduler_step_size LR_SCHEDULER_STEP_SIZE]
               [--tensorboard] [--log_dir LOG_DIR]
               [--output_dir OUTPUT_DIR]
               train_metadata val_metadata label_intent
               label_semiotic label_contextual

positional arguments:
  train_metadata        Path to training metadata
  val_metadata          Path to validation metadata
  label_intent          Path to label for intent
  label_semiotic        Path to label for semiotic
  label_contextual      Path to label for contextual

optional arguments:
  -h, --help            show this help message and exit
  --image_only
  --image_text
  --text_only
  --elmo_image_text
  --elmo_text_only
  --classification CLASSIFICATION
                        Type of loss function to optimize for
  --name NAME           Name of the experiment
  --epochs EPOCHS       Number of epochs
  --lr LR               Learning rate (default is 5e-4 according to the paper)
  --batch_size BATCH_SIZE
                        Batch size
  --shuffle             Whether to shuffle data loader
  --num_workers NUM_WORKERS
                        Number of parallel workers
  --device DEVICE       Pytorch device
  --lr_scheduler_gamma LR_SCHEDULER_GAMMA
                        Decay factor for learning rate scheduler
  --lr_scheduler_step_size LR_SCHEDULER_STEP_SIZE
                        Step size for learning rate scheduler
  --tensorboard
  --log_dir LOG_DIR     Log directory
  --output_dir OUTPUT_DIR
                        Output directory
```

__Specifying the type of inputs__
- ``--image_only`` specifies the code to use only an image as an input.
- ``--image_text`` specifies the code to use both image and caption as inputs. The caption encoder is based on word2vec.
- ``--text_only`` specifies the code to use only a caption as an input. The caption encoder is based on word2vec.
- ``--elmo_image_text`` is similar to ``--image_text`` but using the caption encoder based on ELMo. 
- ``--elmo_text_only`` is similar to ``--text_only`` but using the caption encoder based on ELMo.

__Specifying the type of classification__
- ``--classification intent`` specifies that the optimization task is intent classification.
- ``--classification semiotic`` specifies that the optimization task is semiotic classification.
- ``--classification contextual`` specifies that the optimization task is contextual classification.
- ``--classification all`` specifies that the optimization task is the classification of intent, semiotic, and contextual.
  This optimization task requires the embedding to be suitable for intent, semiotic, and contextual predictions.

## Run all experiments

As our project involves running the same training on different cross-validation splits, different classification tasks, and different type of inputs. We have provided a simple bash script to run all experiments `run_all.sh`. To use it, simply execute
```
./run_all.sh
```
