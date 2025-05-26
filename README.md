# mvt\_fermi

# MVTfermi: Minimum Variability Timescale Analysis for Fermi GBM Data

**MVTfermi** is a Python package designed to calculate the Minimum Variability Timescale (MVT) for Gamma-Ray Bursts (GRBs) observed by the Fermi Gamma-ray Burst Monitor (GBM). It analyzes time-binned light curve data to identify the shortest statistically significant timescales on which source flux changes occur, distinguishing these from Poisson noise.

The core methodology involves analyzing the variance of differenced count rates as a function of data binning resolution, employing Monte Carlo simulations for robust noise characterization, and fitting characteristic curves to identify the MVT. The tool can explore a range of MVT search window widths (\$\Delta\$) and is configured via a YAML file.

## Features

* Calculates MVT from Fermi GBM light curve data.
* Configurable analysis parameters via a YAML file (`config_MVT.yaml`).
* Supports selection of specific detector lists.
* Handles background estimation using user-defined intervals.
* Implements parallel processing for efficient computation.
* Generates diagnostic plots and summary CSV files.
* Provides options for scanning multiple \$\Delta\$ values or performing a binary search for optimal \$\Delta\$.

## Installation

### Prerequisites

* Python >= 3.11
* Git (for installing GDT dependencies directly from GitHub if not installing the pre-built package with resolved dependencies)

### Create and activate a virtual environment

You can use either `venv` (standard Python) or `conda` (recommended for scientific workflows):

#### Option 1: Using `venv` (standard Python)

```bash
python3 -m venv venv_mvtfermi
source venv_mvtfermi/bin/activate  # Linux/macOS
# venv_mvtfermi\Scripts\activate    # Windows
```

#### Option 2: Using `conda`

```bash
conda create -n mvtfermi python=3.11
conda activate mvtfermi
```

### Install the package

#### For general users (standard install from local source):

If you have the `mvt_fermi` package directory (containing `__init__.py` and `pyproject.toml`), navigate to the parent directory and run:

```bash
pip install ./mvt_fermi/
```

If your `pyproject.toml` is at the root of your project (e.g., next to the `mvt_fermi/` folder), just run:

```bash
pip install .
```

#### For developers (editable install from local source):

```bash
pip install -e .
```

#### Once the package is available on PyPI (Future):

```bash
pip install MVTfermi
```

The installation process will handle dependencies listed in `pyproject.toml`, including `astro-gdt` and `astro-gdt-fermi` from their GitHub repositories.

## Configuration

The MVT analysis is primarily controlled by a YAML configuration file, typically named `config_MVT.yaml`. An example structure is:

```yaml
trigger_number: '211211549'       # GRB trigger number
background_intervals: [[-11.67, -1.04], [57.5, 69.58]] # List of [start, end] background time intervals
T90: 34.3046                      # T90 duration of the GRB
det_list: [n2, na]                # List of Fermi GBM detectors to use (e.g., n0-nb for NaIs, b0-b1 for BGOs)
T0: 0                             # Trigger time T0 (seconds)
en_lo: 10                         # Lower energy bound (keV) for data selection
en_hi: 1000                       # Upper energy bound (keV) for data selection
data_path: '/path/to/your/fermi/data/' # Path where GRB data (e.g., TTE files) are located or will be downloaded/cached by GDT
output_path: '/path/to/your/output/'   # Path to save analysis results and plots

## Advance parameters #######
start_padding: 5                  # Factor for extending analysis range before T0 (factor * delt)
end_padding: 5                    # Factor for extending analysis range after T90 (factor * delt)
cores: 2                          # Number of CPU cores for parallel processing
bw: 0.0001                        # Base light curve bin width (seconds) for data processing
delt: 1.0                         # MVT search window width Delta (seconds) for a specific run, or initial for scan
N: 30                             # Number of Monte Carlo iterations for MVT finding
f1: 5                             # Factor for lower bound of fitting range (f1 * MVT_moving_avg_min)
f2: 3                             # Factor for upper bound of fitting range (f2 * MVT_moving_avg_min)
all_delta: False                  # If True, runs for all valid_deltas; otherwise, uses specific delt or binary search logic
all_fig: True                     # If True, generates detailed diagnostic plots for each iteration
```

Place this `config_MVT.yaml` file in the directory from where you will run the `MVTfermi` script. Update paths to reflect your local setup.

## Usage

The package provides a command-line script `MVTfermi` (defined in `pyproject.toml` under `[project.scripts]`).

### Command-line usage

1. Prepare your `config_MVT.yaml` file with the desired parameters and correct paths.
2. Ensure this file is in your current working directory when you run the command.
3. Run the analysis:

```bash
MVTfermi
```

The script will:

* Read `config_MVT.yaml`
* Process the specified GRB data (downloading/caching it via GDT if needed)
* Perform MVT analysis
* Save outputs to the `output_path` specified in the config

> **Important:** Ensure `data_path` and `output_path` in `config_MVT.yaml` point to valid local directories.

## Dependencies

Key dependencies are managed via `pyproject.toml`:

* `numpy`
* `scipy`
* `matplotlib`
* `pandas`
* `PyPDF2`
* `astro-gdt` (from GitHub)
* `astro-gdt-fermi` (from GitHub)
* `termcolor`
* `sigfig`
* `requests`
* `PyYAML` (for config file reading)
* `jupyter` & `notebook` (for development and testing)

See `pyproject.toml` for the complete, version-controlled list.

## License

This project is licensed under the **Apache License 2.0**. See the `license.txt` file for full terms.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests on the [GitHub repository](https://github.com/sumanbala2210-USRA/mvt_fermi).

## Citation

If you use **MVTfermi** in your research, please cite the relevant GDT libraries and any forthcoming publication describing this tool.

> Example: "Please cite Bala et al. (Year) if you use this software in your research."
