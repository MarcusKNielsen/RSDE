# RSDE
Master Thesis Repository

# Structure

MasterThesis/
│
├── C++/                           # Main C++ codebase
│   ├── src/                       # Source code
│   │   ├── main.cpp               # Entry point for your main program
│   │   ├── module1.cpp            # Example C++ module
│   │   ├── module2.cpp            # Additional module
│   │   └── ...                    # More .cpp files
│   │
│   ├── include/                   # Header files
│   │   ├── module1.h              # Header for module1.cpp
│   │   ├── module2.h              # Header for module2.cpp
│   │   └── ...                    # More header files
│   │
│   ├── tests/                     # C++ unit tests
│   │   ├── test_module1.cpp       # Example test
│   │   ├── test_module2.cpp       # Additional test
│   │   └── ...                    # More test files
│   │
│   └── CMakeLists.txt             # CMake configuration file
│
├── Python/                        # Python scripts for experimentation
│   ├── notebooks/                 # Jupyter notebooks
│   │   ├── analysis.ipynb         # Example notebook for data analysis
│   │   └── plotting.ipynb         # Example notebook for plotting
│   │
│   ├── scripts/                   # Python scripts
│   │   ├── experiment1.py         # Experimenting with data or models
│   │   ├── helper_functions.py    # Common utility functions
│   │   └── ...                    # More Python scripts
│   │
│   └── tests/                     # Python unit tests
│       ├── test_experiment1.py    # Example test for experiment1.py
│       └── ...                    # More test files
│
├── data/                          # Dataset storage (if any)
│   ├── raw/                       # Raw data
│   │   └── dataset.csv            # Example raw data file
│   ├── processed/                 # Processed or cleaned data
│   │   └── cleaned_data.csv       # Example processed data file
│   └── README.md                  # Instructions for handling data
│
├── docs/                          # Documentation
│   ├── thesis/                    # Files for the thesis writeup
│   │   ├── thesis.tex             # LaTeX source file
│   │   ├── references.bib         # Bibliography file
│   │   ├── images/                # Figures for the thesis
│   │   └── ...                    # Additional LaTeX resources
│   │
│   └── README.md                  # General project documentation
│
├── results/                       # Results of experiments
│   ├── figures/                   # Plots or visualizations
│   ├── logs/                      # Logs from experiments or simulations
│   └── ...                        # More result files
│
├── .gitignore                     # Git ignore file
├── README.md                      # Overview of the project
├── LICENSE                        # License information (if applicable)
└── requirements.txt               # Python dependencies (for `pip`)