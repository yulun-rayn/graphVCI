# Graph Preparation

Visit our [resource site](https://osf.io/5n2mz/) and download the contents of `graph-prep/graphs` into [graphs](graphs).
  \* *make sure you already have your dataset of interest prepared through `data-prep` branch.*


## Procedure

### 1. Create Initial GRN
Open [GenerateGRN.ipynb](graphs/GenerateGRN.ipynb), set dataset name in the second chunk, then run all.

  \* *if GRN refinement is not desired, the steps below are not necessary.*

### 2. Run Updating Model
```bash
./main.sh &
```

A list of flags may be found in `main.sh` and `src/main.py` for experimentation with different hyperparameters. The models are saved under `*artifact_path*/saves`.

### 3. Generate Updated Adjacency Matrix
Open [UpdateGRN.ipynb](graphs/UpdateGRN.ipynb), set path of the saved model in the first chunk, then run all.

### 4. Generate Updated GRN
Open [GenerateGRN.ipynb](graphs/GenerateGRN.ipynb). Uncomment the third chunk, then run all.

## License

Contributions are welcome! All content here is licensed under the MIT license.
