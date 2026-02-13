# GPU-DBSCAN

A **DBSCAN** implementation in C++/CUDA with performance comparison between **CPU** and **GPU** versions on synthetically generated 2D datasets.

## Overview

The project executes a full pipeline:

1. generate one or more CSV datasets;
2. run DBSCAN on CPU and GPU;
3. validate results (CPU/GPU label assertions);
4. save clustering outputs and benchmark data;
5. visualize results with Python scripts.

The main executable scans datasets in `data_in/`, runs both versions, writes clustering outputs to `data_out/`, and stores benchmark metrics in `benchmark/`.

## Project structure

- `src/main.cu`: CPU/GPU run orchestration, timing, assertions, benchmark I/O.
- `src/cpu/`: CPU DBSCAN implementation.
- `src/gpu/`: CUDA DBSCAN implementation.
- `src/io.*`: CSV parsing/writing and output naming helpers.
- `script/generator.py`: synthetic dataset generator.
- `script/plot.py`: plot a DBSCAN output file.
- `script/benchmark.py`: plot CPU/GPU times and speedup.
- `Makefile`: setup, generation, and plotting helpers.

## Requirements

### C++ Build / Runtime

- C++20 compiler
- CUDA toolkit compatible with your GPU driver
- CMake

### Python scripts

- Python 3.10+
- Dependencies listed in `script/requirements.txt`

Install Python dependencies:

```bash
make init
```

## Build

Example out-of-source build:

```bash
cmake -S . -B build
cmake --build build -j
```

This produces `build/dbscan`.

## Quick workflow

### 1) Generate datasets

Generate a single dataset with default parameters (Makefile defaults: `N=100000`, `C=30`, `CS=1.0`, `STD=0.03`, `NR=0.001`, `R=0`):

```bash
make generate
```

Generate multiple datasets by varying `N` via `N_LIST` (Makefile defaults: `N_LIST="1000 5000 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000"`):

```bash
make multiple-generate
```

You can override parameters from CLI, for example:

```bash
make generate N=500000 C=20 CS=1.5 STD=0.02 NR=0.002 R=42
```

Or generate multiple datasets with different N values:

```bash
make multiple-generate N_LIST="100000 200000 300000"
```

parameters are:
- `N`: number of points for a single dataset to generate
- `N_LIST`: list of N values to generate (used when `make multiple-generate` is called)
- `C`: number of clusters
- `CS`: cluster size
- `STD`: standard deviation
- `NR`: noise ratio
- `R`: random seed

### 2) Run DBSCAN (CPU + GPU)

From the `build` directory:

```bash
./dbscan
```

Expected outputs:

- clustering files in `data_out/` (both CPU and GPU results per dataset);
- benchmark file in `benchmark/` containing timing and speedup.

### 3) Visualize results

Plot a DBSCAN output (`FN` = filename without `.csv`):

```bash
make plot FN=<output_file_name>
```

Plot benchmark results (`BFN` = filename without `.csv`):

```bash
make plot-benchmark BFN=<benchmark_file_name>
```

## Main DBSCAN parameters

Currently defined in `src/common.h`:

- `EPSILON` (default: `0.03`)
- `MIN_PTS` (default: `8`)

To change them, edit these macros and rebuild.

## Data format

### Input datasets (`data_in/`)

CSV with header `x,y`.

### DBSCAN outputs (`data_out/`)

CSV with point coordinates and cluster label (noise/outliers have non-positive labels).

### Benchmark (`benchmark/`)

CSV with columns: dataset name, number of points, CPU time, GPU time, speedup.

## Data cleanup

The target:

```bash
make clean
```

asks for confirmation and then removes `data_in/`, `data_out/` and `benchmark/`.

## Notes

- The project explicitly checks CPU vs GPU clustering results with `assert`; execution fails if labels do not match.
- Input/output paths are defined in `src/common.h` as relative paths (`../data_in/`, `../data_out/`, `../benchmark/`), so run the binary from a directory consistent with those paths (typically `build/`).