# mvt_fermi
# MVTfermi: Minimum Variability Timescale Analysis for Fermi GBM Data

**MVTfermi** is a Python package designed to calculate the Minimum Variability Timescale (MVT) for Gamma-Ray Bursts (GRBs) observed by the Fermi Gamma-ray Burst Monitor (GBM). It analyzes time-binned light curve data to identify the shortest statistically significant timescales on which source flux changes occur, distinguishing these from Poisson noise.

The core methodology involves analyzing the variance of differenced count rates as a function of data binning resolution, employing Monte Carlo simulations for robust noise characterization, and fitting characteristic curves to identify the MVT. The tool can explore a range of MVT search window widths ($\Delta$) and is configured via a YAML file.

## Features

* Calculates MVT from Fermi GBM light curve data.
* Configurable analysis parameters via a YAML file (`config_MVT.yaml`).
* Supports selection of specific detector lists.
* Handles background estimation using user-defined intervals.
* Implements parallel processing for efficient computation.
* Generates diagnostic d and summary CSV files.
* Provides options for scanning multiple $\Delta$ values or performing a binary search for optimal $\Delta$.

## Installation

### Prerequisites

* Python >= 3.11
* Git (for installing GDT dependencies directly from GitHub if not installing the pre-built package with resolved dependencies)

### Steps

1.  **Clone the repository (Optional, if installing from a local source copy):**
    ```bash
    git clone https://github.com/sumanbala2210-USRA/mvt_fermi # Replace with your actual repo URL
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv_mvtfermi
    source venv_mvtfermi/bin/activate  # On Linux/macOS
    # venv_mvtfermi\Scripts\activate    # On Windows
    ```

3.  **Install the package:**

    * **For general users (standard install from local source):**
        If you have the `mvt_fermi` package directory (the one containing `__init__.py`, `pyproject.toml` at its root or one level up), navigate to the directory *containing* the `mvt_fermi` folder (e.g., `mvt_fermi_project` if your structure is `mvt_fermi_project/mvt_fermi/`) and run:
        ```bash
        pip install ./mvt_fermi/ 
        ```
        (Note: If `pyproject.toml` is inside `mvt_fermi/`, then from the parent directory you'd do `pip install ./mvt_fermi/`. If `pyproject.toml` is in the root alongside the `mvt_fermi/` source folder, you'd run `pip install .` from the root.)
        *Assuming your `pyproject.toml` is in the root of `mvt_fermi_project` and your package source code is in `mvt_fermi_project/mvt_fermi/`, then from within `mvt_fermi_project` you would typically run `pip install .`.*

        Given your instruction `pip install mvt_fermi/`, it implies you are one level above the `mvt_fermi` source directory.

    * **For developers (editable install from local source):**
        Navigate to the directory that *contains* your `mvt_fermi` source package directory (i.e., the root of your project where `pyproject.toml` is located). For example, if your structure is `project_root/mvt_fermi/`, you run this from `project_root/`:
        ```bash
        pip install -e ./mvt_fermi/
        ```
        *Similar to above, if `pyproject.toml` is in `project_root/` and your package code is in `project_root/mvt_fermi/`, the command from `project_root/` is `pip install -e .` if `pyproject.toml` specifies `mvt_fermi` as the package, or `pip install -e mvt_fermi` if `pyproject.toml` is set up to find packages itself. The most standard for a project root containing `pyproject.toml` and a source folder like `mvt_fermi` is just `pip install -e .` from the root.*

    * **Once the package is available on PyPI (Future):**
        ```bash
        pip install MVTfermi 
        ```
        *(Note: The name on PyPI will match the `name` field in your `pyproject.toml`)*

    The installation process will handle the dependencies listed in `pyproject.toml`, including `astro-gdt` and `astro-gdt-fermi` from their GitHub repositories.

## Configuration

The MVT analysis is primarily controlled by a YAML configuration file, typically named `config_MVT.yaml`. An example structure is:

```yaml
trigger_number: '211211549'       # GRB trigger number
background_intervals: [[-11.67, -1.04], [57.5, 69.58]] # List of [start, end] background time intervals
T90: 34.3046                    # T90 duration of the GRB
det_list: [n2, na]                # List of Fermi GBM detectors to use (e.g., n0-nb for NaIs, b0-b1 for BGOs)
T0: 0                             # Trigger time T0 (seconds)
en_lo: 10                         # Lower energy bound (keV) for data selection
en_hi: 1000                       # Upper energy bound (keV) for data selection
data_path: '/path/to/your/fermi/data/' # Path where GRB data (e.g., TTE files) are located or will be downloaded/cached by GDT
output_path: '/path/to/your/output/'   # Path to save analysis results and d
## Advance parameters #######
start_padding: 5                  # Factor for extending analysis range before T0 (factor * delt)
end_padding: 5                    # Factor for extending analysis range after T90 (factor * delt)
cores: 2                          # Number of CPU cores for parallel processing
bw: 0.0001                        # Base light curve bin width (seconds) for data processing
delt: 1.0                         # MVT search window width Delta (seconds) for a specific run, or initial for scan
N: 30                             # Number of Monte Carlo iterations for MVT finding
f1: 5                             # Factor for lower bound of fitting range (f1 * MVT_moving_avg_min)
f2: 3                             # Factor for upper bound of fitting range (f2 * MVT_moving_avg_min)
all_delta: False                  # If True, runs for all valid_deltas; otherwise, uses specific delt or binary search logic (as per main script's logic)
all_fig: True                     # If True, generates detailed diagnostic d for each iteration (controlled by MVTAnalyzer)



Place this config_MVT.yaml file in the directory from where you will run the MVTfermi script. Update paths in the YAML file to point to your data and desired output locations.

Usage
The package provides a command-line script MVTfermi (as defined in your pyproject.toml [project.scripts]).

Prepare your config_MVT.yaml file with the desired parameters and correct paths.
Run the analysis: Open your terminal, activate the virtual environment (if you used one), and navigate to the directory containing your config_MVT.yaml. Then execute:
Bash
MVTfermi
The script will read config_MVT.yaml, process the specified GRB data (potentially downloading/caching it via the GDT libraries), perform the MVT analysis, and save outputs to the specified output_path.



Important: Update data_path and output_path in your config_MVT.yaml to valid locations on your system.

Usage
The package provides a command-line script MVTfermi (as defined in your pyproject.toml [project.scripts]).

Prepare your config_MVT.yaml file with the desired parameters and correct paths. Ensure this file is in your current working directory when you run the command.
Run the analysis: Open your terminal, activate the virtual environment (if you used one), and ensure config_MVT.yaml is present. Then execute:
Bash
MVTfermi
The script will read config_MVT.yaml, process the specified GRB data (potentially downloading/caching it via the GDT libraries), perform the MVT analysis, and save outputs to the output_path specified in the config file.
Programmatic Usage (Conceptual):

The primary intended use is via the MVTfermi command-line script. However, if the core analysis routines (like the MVTAnalyzer class if you integrate it, or specific functions) are exposed through the package's __init__.py, they could be imported and used in Python scripts or Jupyter notebooks.


Dependencies
Key dependencies are managed by pip via the pyproject.toml file:

numpy
scipy
matplotlib
pandas
PyPDF2
astro-gdt (from GitHub)
astro-gdt-fermi (from GitHub)
termcolor
sigfig
requests
PyYAML (for reading config_MVT.yaml)
jupyter & notebook (listed as dependencies, mainly for development/testing environments)
See pyproject.toml for the complete and versioned list.

License
This project is licensed under the Apache License 2.0. See the license.txt file for details.
(Ensure your license.txt file actually contains the Apache 2.0 license text).

Contributing
Contributions are welcome! Please feel free to submit issues or pull requests on the GitHub repository.

Citation
If you use MVTfermi in your research, please describe the methodology used and consider citing the relevant GDT packages and any forthcoming paper describing this tool.
(Once you have a paper, you can add: "Please cite [Your Paper Details, e.g., Bala et al. (Year)] if you use this software in your research.")








# MVTfermi: Minimum Variability Timescale Analysis for Fermi GBM Data

**MVTfermi** is a Python package designed to calculate the Minimum Variability Timescale (MVT) for Gamma-Ray Bursts (GRBs) observed by the Fermi Gamma-ray Burst Monitor (GBM). It analyzes time-binned light curve data to identify the shortest statistically significant timescales on which source flux changes occur, distinguishing these from Poisson noise.

The core methodology involves analyzing the variance of differenced count rates as a function of data binning resolution, employing Monte Carlo simulations for robust noise characterization, and fitting characteristic curves to identify the MVT. The tool can explore a range of MVT search window widths ($\Delta$) and is configured via a YAML file.

---

## Features

* Calculates MVT from Fermi GBM light curve data
* Configurable analysis via a YAML file (`config_MVT.yaml`)
* Detector-specific analysis
* Background estimation with user-defined intervals
* Parallel processing support
* Generates diagnostic plots and CSV summaries
* Scans multiple $\Delta$ values or performs binary search

---

## Installation

### Prerequisites

* Python >= 3.11
* Git (for some dependencies if not using a pre-built package)

### Steps

1. **Clone the repository (optional):**

```bash
git clone https://github.com/sumanbala2210-USRA/mvt_fermi
```

2. **Create and activate a virtual environment:**

```bash
python3 -m venv venv_mvtfermi
source venv_mvtfermi/bin/activate  # Linux/macOS
# venv_mvtfermi\Scripts\activate    # Windows
```

3. **Install the package:**

* **From local source:**

```bash
pip install ./mvt_fermi/
```

* **Editable (developer) install:**

```bash
pip install -e ./mvt_fermi/
```

* **Future PyPI release:**

```bash
pip install MVTfermi
```

Dependencies such as `astro-gdt` and `astro-gdt-fermi` will be automatically handled via `pyproject.toml`.

---

## Configuration

Create a `config_MVT.yaml` file to control the analysis:

```yaml
trigger_number: '211211549'       # GRB trigger number
background_intervals: [[-11.67, -1.04], [57.5, 69.58]] # List of [start, end] background time intervals
T90: 34.3046                    # T90 duration of the GRB
det_list: [n2, na]                # List of Fermi GBM detectors to use (e.g., n0-nb for NaIs, b0-b1 for BGOs)
T0: 0                             # Trigger time T0 (seconds)
en_lo: 10                         # Lower energy bound (keV) for data selection
en_hi: 1000                       # Upper energy bound (keV) for data selection
data_path: '/path/to/your/fermi/data/' # Path where GRB data (e.g., TTE files) are located or will be downloaded/cached by GDT
output_path: '/path/to/your/output/'   # Path to save analysis results and d
## Advance parameters #######
start_padding: 5                  # Factor for extending analysis range before T0 (factor * delt)
end_padding: 5                    # Factor for extending analysis range after T90 (factor * delt)
cores: 2                          # Number of CPU cores for parallel processing
bw: 0.0001                        # Base light curve bin width (seconds) for data processing
delt: 1.0                         # MVT search window width Delta (seconds) for a specific run, or initial for scan
N: 30                             # Number of Monte Carlo iterations for MVT finding
f1: 5                             # Factor for lower bound of fitting range (f1 * MVT_moving_avg_min)
f2: 3                             # Factor for upper bound of fitting range (f2 * MVT_moving_avg_min)
all_delta: False                  # If True, runs for all valid_deltas; otherwise, uses specific delt or binary search logic (as per main script's logic)
all_fig: True                     # If True, generates detailed diagnostic d for each iteration (controlled by MVTAnalyzer)
```

> Place this file in the directory where you will run the script and update the `data_path` and `output_path` accordingly.

---

## Usage

### CLI Usage

Ensure your `config_MVT.yaml` file is present in the current directory.

```bash
MVTfermi
```

This will:

* Load configuration from the YAML file
* Download/process Fermi GBM data (via `astro-gdt` if needed)
* Perform the MVT analysis
* Save outputs to the configured `output_path`

### Programmatic Usage (Optional)

If you expose functionality in `__init__.py`, you can use classes like `MVTAnalyzer` directly in Python scripts or notebooks:

```python
from mvt_fermi import MVTAnalyzer
analyzer = MVTAnalyzer(config_path='config_MVT.yaml')
analyzer.run()
```

---

## Dependencies

Managed via `pyproject.toml`:

* numpy
* scipy
* matplotlib
* pandas
* PyPDF2
* astro-gdt (GitHub)
* astro-gdt-fermi (GitHub)
* termcolor
* sigfig
* requests
* PyYAML
* jupyter, notebook (for development)

---

## License

Licensed under the Apache License 2.0. See `license.txt` for details.

---

## Contributing

Contributions welcome! Submit issues or pull requests via the GitHub repository.

---

## Citation

If you use **MVTfermi** in research, describe the methodology and cite the relevant GDT packages. Once a formal publication is available, please cite it accordingly.

> Example: “Please cite Bala et al. (Year) if you use this software.”
