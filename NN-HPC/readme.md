This Readme should describe each Folder:


- Base: contains abstract interfaces and shared logics, like base_model.py and base_data_loader.py


configs/:
Purpose: All experiment settings live here; code should be as parameter-free as possible.


data_loaders/

Purpose: Reading data from source and handing it to the pipeline.


models/

Purpose: Model architectures / builders.


optimizers/

Purpose: Factory for optimizers and learning-rate policies.


utils/

Purpose: Everything “miscellaneous” and reusable.


wrappers/

Purpose: Adapt components to other ecosystems (scikit-learn, CLI, etc.).


Inspiration from: https://github.com/Python-templates/sklearn-project-template


