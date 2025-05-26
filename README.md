
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
