# Project structure
- data
  - data_hm.txt: Dataset with alphabetic sequence for inital model training testing
  - data.txt: the shakespeare dataset
- plots: plot s of cross entropy and perplexity evolution during training  
- project_code
  - models: parameters of all pretrained models
  - output: generated texts by trained models (these models can be found in models folder)
  - custum_dataset.py: built for dataloader 
  - main.ipynb: main code for training. testing and generating text
  - model.py: implementation of model
  - pretrained_model_generate.ipynb: generate text with all pretrained models in models folder
  - util.py: utility functions
  
  # Train a new model, test it and generate text
  Execute main.ipynb
  # Generate new texts with pretrained models
  The pretrained models can be found in [Drive](https://drive.google.com/drive/folders/1BZmO2twE9XscTUtDsi9No0Ua50h47A1h?usp=sharing)

  
  Execute pretrained_model_generate.ipynb
  