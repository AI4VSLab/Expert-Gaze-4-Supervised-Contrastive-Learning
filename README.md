# Folder Structure 


- **dataset** : all code related to datasets
    - datamodule: pl.LightningDataModule objects for managing data
    - dataset: torch Dataset objects 
    - transforms: contains custom transforms
    - utils: helper functions
- **metrics**: files for computing metrics
- **nn**: model definitions and loss functions
    - loss: loss functions
    - nn: actual mdoel definitions
    - optimizers
    - pl_modules: LightningModules for all of our models
- **run**: all bash and config files used to run models
    - egd_mimic: egd_mimic dataset
    - gaze_reports: running image models with pseudo labels from gaze
    - GazeFormerMD: running GazeFormerMD
    - wsupcon_unseen_dataset: running models on unseen dataset
    - utils: helper functions which include custom parser





More Coming Soon!