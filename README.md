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

The **MVTfermi** package provides two equivalent ways to run the analysis:

1. **Terminal-based interface:** using the `MVTfermi` command-line script
2. **Python/Jupyter interface:** using the `mvtgeneral()` function

Both methods share the same logic and support the same options (including optional `delta`).

---

### 1. Terminal usage: `MVTfermi` and `MVTgeneral`

```bash
MVTfermi
```

Optionally, specify a delta value:

```bash
MVTfermi --delta 0.5     # run with delta=0.5
MVTfermi --delta all     # scan all valid delta values
```

This command will:

* Read `config_MVT.yaml`
* Optionally read cached `.npz` data if it exists
* Run the full MVT analysis
* Save plots and CSV results in the specified output folder


```bash
MVTgeneral                 # Runs with config_MVT_general.yaml, requires .npz data
MVTgeneral --delta 0.5
MVTgeneral --delta all
```

This command will:

* Read `config_MVT.yaml`
* Read cached `.npz` data (REQUIRED!!)
* Run the full MVT analysis
* Save plots and CSV results in the specified output folder
---

### 2. Python or Jupyter usage: `mvtfermi()` `mvtgeneral()`


Import and run `mvtfermi()` directly:

```python
from MVTfermi.mvt_parallel_fermi import mvtfermi
```

You can call it in two ways:

#### a. Auto-read config and data

```python
mvtfermi()
mvtfermi(delta="all")      # Scan all delta values (from valid_deltas list)
mvtfermi(delta=0.5)        # Run with fixed delta=0.5
```

* Load settings from `config_MVT.yaml`
* Load `.npz`-cached arrays if available
* Run the analysis using provided or default `delta`


Import and run `mvtgeneral()` directly:

```python
from MVTfermi.mvt_parallel_general import mvtgeneral
```

You can call it in two ways:

#### a. Auto-read config and data

```python
mvtgeneral(delta="all")      # Scan all delta values (from valid_deltas list)
mvtgeneral(delta=0.5)        # Run with fixed delta=0.5
```

If no arrays are passed, it will:

* Load settings from `config_MVT.yaml`
* Load `.npz`-cached arrays if available
* Run the analysis using provided or default `delta`

#### b. Custom light curve arrays

```python
mvtgeneral(time_edges=my_edges, counts=my_counts, back_counts=my_background, delta=0.25)
```

This allows you to skip config/data file loading and analyze your own in-memory data.

---

# MVTfermi & MVTgeneral — CLI and Python Usage Overview

| Tool / Interface [`Yaml` used]         | Type         | Accepts delta input?        | Accepts arrays?        | .npz data usage           | Description                          |
|---------------------------------------|--------------|----------------------------|-----------------------|---------------------------|------------------------------------|
| `MVTfermi` [`config_MVT`]              | Terminal     | ✅ Yes (`--delta`)           | ❌ No                  | Optional, cached if exists | Terminal tool for standard MVT analysis |
| `MVTgeneral` [`config_MVT_general`]    | Terminal     | ✅ Yes (`--delta`)           | ❌ No                  | ⚠️ Requires .npz data file  | Terminal tool requiring `.npz` light curve data |
| `mvtfermi()` [`config_MVT`]             | Python func  | ✅ Yes (`delta=`)            | ❌ No                  | Optional `.npz` or arrays  | Python API, can auto-read config    |
| `mvtgeneral()` [`config_MVT_general`]   | Python func  | ✅ Yes (`delta=`)            | ⚠️ Must if no `.npz`     | ⚠️ Must if no array given   | Python API, flexible data input     |


<table>
  <thead>
    <tr>
      <th style="min-width: 180px;">Tool / Interface <br><code>[Yaml used]</code></th>
      <th style="min-width: 120px;">Type</th>
      <th style="min-width: 360px;">Accepts delta input?</th>
      <th style="min-width: 180px;">Accepts arrays?</th>
      <th style="min-width: 130px;">.npz data usage</th>
      <th style="min-width: 130px;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>MVTfermi</code><br><small>[config_MVT]</small></td>
      <td>Terminal</td>
      <td>✅ Yes (<code>--delta</code>)</td>
      <td>❌ No</td>
      <td>Optional, cached if exists</td>
      <td>Terminal tool for standard MVT analysis</td>
    </tr>
    <tr>
      <td><code>MVTgeneral</code><br><small>[config_MVT_general]</small></td>
      <td>Terminal</td>
      <td>✅ Yes (<code>--delta</code>)</td>
      <td>❌ No</td>
      <td><span style="background-color: #fff3cd;">⚠️ Requires .npz data file</span></td>
      <td>Terminal tool requiring <code>.npz</code> light curve data</td>
    </tr>
    <tr>
      <td><code>mvtfermi()</code><br><small>[config_MVT]</small></td>
      <td>Python function</td>
      <td>✅ Yes (<code>delta=</code>)</td>
      <td>❌ No</td>
      <td>Optional <code>.npz</code> or arrays</td>
      <td>Python API, can auto-read config</td>
    </tr>
    <tr>
      <td><code>mvtgeneral()</code><br><small>[config_MVT_general]</small></td>
      <td>Python function</td>
      <td>✅ Yes (<code>delta=</code>)</td>
      <td><span style="background-color: #fff3cd;">⚠️ Must if no <code>.npz</code></span></td>
      <td><span style="background-color: #fff3cd;">⚠️ Must if no array given</span></td>
      <td>Python API, flexible data input</td>
    </tr>
  </tbody>
</table>




---




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
* `PyYAML`
* `jupyter` & `notebook`

See `pyproject.toml` for the complete, version-controlled list.

## License

This project is licensed under the **Apache License 2.0**. See the `license.txt` file for full terms.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests on the [GitHub repository](https://github.com/sumanbala2210-USRA/mvt_fermi).

## Citation

If you use **MVTfermi** in your research, please cite the relevant GDT libraries and any forthcoming publication describing this tool.

> Example: "Please cite Bala et al. (Year) if you use this software in your research."
