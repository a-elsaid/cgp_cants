# GP-CANTS

GP-CANTS (Genetic Programming CANTS) is a nature-inspired metaheuristic framework for graph-based genetic programs, designed to automate neural architecture search (NAS) and produce interpretable computational graphs. Combining ant colony optimization (CANTS) with genetic programming (GP) and optional backpropagation, GP-CANTS enables multi-colony, multi-agent exploration of an unbounded continuous search space, yielding adaptive and scalable architectures.

## About This Work
Based on the soon-to-be-published journal article “CANTS-GP: A Nature-Inspired Metaheuristic for Graph-Based Genetic Programs” (ElSaid & Desell) and accepted as a poster at the GECCO 2025 conference, GP-CANTS introduces several key innovations:

  - **Multi-Colony Evolution:** Parallel ant colonies evolve both at the agent level (ant movement strategies and pheromone control) and at the colony level (parameter optimization via PSO).
  - **Continuous 3D Search Space:** Ant agents traverse a three-dimensional continuous space, with paths clustered by DBSCAN to form graph nodes and DFS-based cycle removal ensuring valid computational graphs.
  - **Graphical Genetic Programs:** Nodes represent GP functions (e.g., addition, multiplication, trigonometric functions) with trainable weights, supporting richer, more flexible structures than traditional neural cells.
  - **Adaptive Pheromone Control:** Dynamic evaporation rates and ant mortality tuned per colony to balance exploration and exploitation throughout the search.
  - **Interpretable, High-Performance Models:** Demonstrated on aviation, power-plant, and wind-turbine time-series benchmarks, CANTS-GP outperforms state-of-the-art GP and NAS methods while producing concise, human-readable equations.

## Features
  - **Graph-Based Representation**: Model neural architectures as dynamic graphs.  
  - **Ant Colony Optimization (CANTS)**: Leverage multi-agent exploration.  
  - **Backpropagation**: Improve solutions with gradient-based learning.  
  - **Parallel Execution with MPI:** Distribute multiple colonies across processes using MPI for scalable, concurrent search.
  - **Multithreaded Training/Evaluation:** Utilize Python threading within each MPI process to train and assess candidate graphs in parallel.
  - **Configurable Experiments:** Support for multiple datasets, normalization, and logging levels.
  - **Logging & Visualization:** Built-in logging via Loguru and graph visualizations using Graphviz.



## Prerequisites

- **Python**: Version 3.8 or higher.  
- **System Dependencies:**
    - Graphviz (CLI & dev tools) for visualization
        - Debian/Ubuntu: `sudo apt-get install graphviz`
        - macOS: `brew install graphviz`
    - MPI (OpenMPI or MPICH) for distributed execution
        - Debian/Ubuntu: sudo apt-get install openmpi-bin libopenmpi-dev
        - macOS: brew install open-mpi
- **Python Packages**:  
  ```bash
  pip install numpy pandas scikit-learn torch graphviz loguru matplotlib mpi4py ipdb
  ```
  

## Repository Structure


    ├── ant.py            # Ant agent implementation
    ├── colony.py         # Single-colony class
    ├── colonies.py       # Main entry for multi-colony experiments
    ├── graph.py          # Graph structure & NAS representation
    ├── helper.py         # Argument parsing & setup utilities
    ├── node.py           # Graph node definition
    ├── search_space.py   # Parameter and search-space definitions
    ├── timeseries.py     # Time-series data loader
    ├── util.py           # Utility functions (e.g., center-of-mass)
    ├── tests.py          # Unit tests and usage examples
    └── run.sh            # Example Bash script to configure and run experiments


## Usage
### 1. Configure run.sh

This script uses a set of bash arrays and variables to parameterize your experiments. In particular:

- `DATA_DIRS[<key>]` maps each dataset key (e.g., wind, c172) to its directory path.
- `INPUT[<key>]` lists the space-separated feature names used as inputs for that dataset.
- `OUTPUT[<key>]` specifies the target variable(s) for prediction.
- `FILE_NAMES[<key>]` enumerates the CSV filenames to be processed.

You can also adjust top‑level variables like LOG_DIR, OUT_DIR, LIVING_TIME, and MPI settings (e.g., -np for mpirun) directly at the top of run.sh to control logging locations, output folders, runtime duration, and distributed execution parameters.

Edit the arrays in run.sh to point to your local datasets, input/output variable names, and file names. For example:

  ```bash
    # Set dataset directories
    DATA_DIRS[wind]="/path/to/your/wind_dataset"
    DATA_DIRS[c172]="/path/to/your/c172_dataset"
    
    # Define input features (space-separated)
    INPUT[wind]="AltAGL AltB AltGPS AltMSL BaroA E1_CHT2 ..."
    # Define output target(s)
    OUTPUT[wind]="E1_CHT1"
    # List the data file names in each directory
    FILE_NAMES[wind]="wind1.csv wind2.csv"
```
## 2. Run the Experiments

Simply execute:

```bash
  bash run.sh
```

This will launch the configured experiments in sequence, saving logs to the specified LOG directory and outputs to the specified OUT directory.

## 3. Command-Line Options

The main script supports the following options (many correspond to the arrays and variables in run.sh):

- `--data_files:` list of CSV files to process (mirrors FILE_NAMES entries in run.sh).
- `--input_names:` space-separated input feature names (mirrors INPUT entries).
- `--output_names:` target variable(s) for prediction (mirrors OUTPUT entries).
- `--data_dir:` base directory for your datasets (mirrors DATA_DIRS).
- `--log_dir:` directory where logs will be saved.
- `--out_dir:` directory for output files (graphs, results, etc.).
- `--living_time:` number of PSO iterations (how long each colony lives).
- `--use_bp:` enable backpropagation for weight training (future versions may support a BP-free CANTS mode).
- `--bp_epochs:` number of backpropagation epochs when --use_bp is set.
- `--loss_fun:` loss function to optimize (default: mse for regression problems).
- `--comm_interval:` iteration interval for colonies to exchange information via PSO.

Example invocation:

```bash
python colonies.py \
  --data_files file1.csv file2.csv \
  --input_names AltAGL AltB ... \
  --output_names E1_CHT1 \
  --data_dir /path/to/datasets \
  --log_dir logs \
  --out_dir out \
  --living_time 90 \
  --use_bp \
  --bp_epochs 10 \
  --loss_fun mse \
  --comm_interval 5
```

## 4. Parallel & Distributed Execution. Parallel & Distributed Execution

For large-scale or multi-node setups, leverage MPI to run colonies in parallel:

```bash
mpirun -np <num_processes> python colonies.py [args]
```


An MPI process will work as the environment organizing the colonies PSO search. Each of the other MPI processes will run one colony, generating candidate graphs, which are trained and evaluated concurrently using Python threads.

**Contact**  
For questions, bug reports, or contributions, please reach out to:  

A. Ahmed ElSaid, Ph.D.  
Assistant Professor, Computer Science  
University of North Carolina at Wilmington  
Email: [elsaida@uncw.edu](mailto:elsaida@uncw.edu)  
Web: [https://aelsaid.net](https://aelsaid.net)  
