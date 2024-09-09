This code recreates the figures from [Segmental retention models for representing the hydraulic properties of evolving structured soils](https://www.doi.org/10.1002/vzj2.20378). 
It consists of the following files

* `README.md` -- this file.
* `LICENSE.txt` -- text of the MIT licence, applicable to the code (not to the data) contained in this repository.
* `ruff.toml` -- configuration file for the Ruff Python formatter.
* `segmental_models.yaml` -- specification of the Python packages necessary to run the code, in YAML format.
* `vangenuchten.py` -- a simple Python implementation of the [Van Genuchten soil water retention curve](https://doi.org/10.2136/sssaj1980.03615995004400050002x).
* `MJGCB2020_PoSD.csv` -- A comma-separated values file containing the time-dependent porosities for the compaction example, based on [A framework for modelling soil structure dynamics induced by biological activity](https://www.doi.org/10.1111/gcb.15289).
* `MJB2020_PoSD.csv` -- As above, but for the organic amendment example, based on [Modelling dynamic interactions between soil structure and the storage and turnover of soil organic matter](https://www.doi.org/10.5194/bg-17-5025-2020).
* `segmental_models.py` -- the main Python file implementing the segmental retention models and the conductivity estimate.
* `segmental_models.ipynb` -- A Jupyter notebook recreating Figs. 1 and 2.
* `segmental_models_compaction.ipynb` -- A Jupyter notebook recreating Fig. 3.
* `segmental_models_amendment.ipynb` -- A Jupyter notebook recreating Fig. 4.

Follow the instructions below to set up and run the code.

### Download or clone the repository

To download the latest version, click the green `<> Code` button and `Download ZIP`.

If you're using git, you can alternatively clone the entire repository using

```bash
git clone <url of this repository> segmental-retention-models && cd segmental-retention-models
```

### Set up the environment

If you don't have `conda` or `mamba` installed, install the [Miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file) distribution following the instructions in its readme.
Then create and activate the environment using

```bash
mamba env create -f segmental_models.yaml && conda activate segmental_models
```

### Start the notebook server

Start the notebook server

```bash
jupyter notebook
```

In the browser window that opens, open the `.ipynb` files and use `Run All Cells` from the `Run` menu to run them.
