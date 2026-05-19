# EQW Stellar Spectra Measurement

Interactive Python tool for measuring **equivalent widths (EQW)** of absorption lines in stellar spectra. Fits Gaussian profiles to spectral lines and exports results in **MOOG-compatible format** for stellar abundance analysis.

Originally written in SuperMongo, rewritten and corrected in Python.

---

## Features

- Reads 1D stellar spectra in **FITS format**
- Supports line lists for **Fe** and **other elements**
- Interactive line-by-line measurement with adjustable continuum and capture range
- Iterative **non-linear Gaussian fitting** (least squares)
- **Sigma rejection** step to discard poorly fitted lines
- Output in **MOOG format** ready for abundance analysis

---

## Requirements

- Python 3.x
- [Anaconda](https://www.anaconda.com/) recommended

### Dependencies

```bash
pip install PyAstronomy
```

The following packages are included with Anaconda and require no additional installation:

```
numpy
pandas
matplotlib
```

---

## Input Files

| File | Description |
|------|-------------|
| `*.fits` | 1D stellar spectrum in FITS format (wavelength in Ångströms) |
| `linelist_fe.dat` | Line list for Fe lines |
| `linelist_other.dat` | Line list for other elements |

### Line list format

Whitespace-separated, with a header row:

```
lambda   el    ep     loggf   damp
5166.282 26.0  0.000  -4.195  0.279
5168.897 26.0  1.485  -1.100  0.270
...
```

Columns: wavelength (Å) | element number | excitation potential (eV) | log(gf) | damping constant

---

## Usage

1. Place your `.fits` file and line list files in the same folder as the script.
2. The script will ask for the FITS filename at startup:

```python
file_name_spectra_fit = input('Enter the FITS filename (e.g. my_spectrum.fits): ').strip()
filename_line_list_Fe    = 'linelist_fe.dat'
filename_line_list_other = 'linelist_other.dat'
```

3. Run from the Anaconda Prompt (or any terminal with the environment active):

```bash
python eqw_medicion_corregido.py
```

4. Follow the interactive prompts:
   - Choose line list (Fe or other elements)
   - For each line: adjust capture range and continuum level
   - Accept or discard each measurement
   - Optionally apply sigma rejection at the end

---

## Output Files

| File | Description |
|------|-------------|
| `<name>.Fe.eqw.dat` | All valid EQW measurements |
| `<name>.Fe.MOOG.dat` | Final measurements after sigma rejection, sorted by element, ready for MOOG |

Output format (MOOG-compatible):

```
lambda   el   ep     loggf   damp   disp   EQW(mÅ)
5166.282 26.0 0.000  -4.195  0.279  0.0    95.3
```

---

## Workflow

```
Load FITS spectrum
       │
       ▼
Filter line list to spectrum range
       │
       ▼
For each line:
  ├─ Adjust continuum & capture range
  ├─ Fit Gaussian (iterative least squares)
  ├─ Plot fit interactively
  └─ Accept / discard / repeat
       │
       ▼
Save .eqw.dat
       │
       ▼
Sigma rejection (optional)
  └─ Polynomial fit to σ vs λ
  └─ Reject outliers
       │
       ▼
Save .MOOG.dat
```

---

## Known Limitations / Future Work

- The measurement process is fully manual (one line at a time)
- No mid-session save/resume: the script must be completed in one run
- EQW integration uses a fixed 0.001 Å step over the fitted Gaussian rather than the actual pixel grid

These are planned improvements for a future version.

---

## License

MIT License — free to use and modify.
