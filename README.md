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
- **System**:  
  - [Graphviz](https://graphviz.org/) (install via `apt`, `brew`, or your package manager).  
- **Python Packages**:  
  ```bash
  pip install numpy pandas scikit-learn torch graphviz loguru matplotlib
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

Edit the arrays in run.sh to point to your local datasets, input/output variable names, and file names. For example:
  
  ```
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

**Contact**  
For questions, bug reports, or contributions, please reach out to:  

A. Ahmed ElSaid, Ph.D.  
Assistant Professor, Computer Science  
University of North Carolina at Wilmington  
Email: [elsaida@uncw.edu](mailto:elsaida@uncw.edu)  
Web: [https://aelsaid.net](https://aelsaid.net)  
