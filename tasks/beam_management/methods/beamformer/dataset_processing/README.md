# BeamFormer-Dataset

A MATLAB toolbox for generating **MIMO channel state information (CSI) datasets** from ray-tracing simulation outputs. The pipeline reads ray-tracing results stored in SQLite databases, applies antenna array geometry, optional antenna pattern corrections, and exports CSI matrices as `.mat` files suitable for machine learning research on channel prediction.

---

## Overview

The tool converts raw ray-tracing path data (angles, delays, complex field components) into complex-valued MIMO CSI tensors. The intended use case is the `homeoffice-communication` scenario at **28 GHz** with a **16×16 Tx** and **2×1 Rx** array configuration.

**Processing pipeline:**

```
SQLite ray-tracing data
        ↓
  read_sqlite()         — load path geometry, UTD coefficients
        ↓
  get_valid_channels()  — identify Tx–Rx channel pairs
        ↓
  generate_tilt_info()  — compute beam steering angles
        ↓
  generate_array()      — build Tx/Rx antenna array coordinates
        ↓
  fetch_*_data()        — extract polarization & MIMO phase data
        ↓
  synthesis_elec_data() — combine E-field components (V+H polarization)
        ↓
  siso2mimo()           — expand SISO to MIMO via array steering
        ↓
  take_pattern_influence() — (optional) apply patch antenna gain pattern
        ↓
  CSI .mat output       + SQLite meta-info record
```

---

## Repository Structure

```
BeamFormer-Dataset/
├── README.md
├── main.m                      # Main entry point
├── config.m                    # Configuration (28G, homeoffice-communication scenario)
├── <csi_output_folder><suffix>/            # Output directory for generated CSI files (auto-created)
│       ├── t<M>x<N>_r<M>x<N>_csi/             # CSI .mat files
│       ├── t<M>x<N>_r<M>x<N>_metainfo/        # Per-run angle metadata (.sqlite)
│       └── t<M>x<N>_r<M>x<N>_metainfo_detail/ # Per-sample detail (.csv)
└── utils/
    ├── read_sqlite.m               # Load ray-tracing SQLite databases
    ├── list_paths.m                # Enumerate input SQLite folders
    ├── get_valid_channels.m        # Identify valid Tx–Rx pairs
    ├── generate_tilt_info.m        # Compute beam steering / tilt angles
    ├── generate_array.m            # Build 2D antenna array coordinates
    ├── generate_match_tables.m     # Match path IDs across V/H polarization
    ├── fetch_elec_data_for_match_table.m  # Extract UTD E-field data
    ├── fetch_mimo_data_for_match_table.m  # Compute MIMO phase / AoA / AoD
    ├── synthesis_elec_data.m       # Combine V+H polarization components
    ├── siso2mimo.m                 # Expand SISO path to MIMO via array steering
    ├── take_pattern_influence.m    # Apply antenna gain pattern
    ├── generate_patch_antenna.m    # Load/configure patch antenna model
    ├── load_and_preprocess_uan.m   # Parse .uan antenna pattern files
    ├── modify_antenna_gain.m       # Scale antenna gain data
    ├── calculate_rotation_matrix.m # Rotation matrix utilities
    ├── filter_dirty_paths.m        # Remove invalid ray-tracing paths
    ├── check_estimate_angles.m     # Validate AoA/AoD estimates
    ├── write_csi_angle_meta_info.m # Write angle metadata to SQLite
    ├── save_meta_record.m          # Save per-sample CSV metadata
    ├── convertStructToGpu.m        # Move struct arrays to GPU
    ├── isMatrixOnGPU.m             # Check if matrix is on GPU
    ├── MicroPatchInfo.m            # Patch antenna geometry constants
    ├── progressBar.m               # Console progress display
    ├── half-wave-dipole.uan        # Dipole antenna pattern file
    └── patch.uan                   # Patch antenna pattern file
```

---

## Requirements

- MATLAB R2021b or later
- MATLAB Database Toolbox (for `sqlite()`)
- (Optional) Parallel Computing Toolbox (for GPU acceleration)

---

## Input Data Format

The raw ray-tracing SQLite dataset for the **28 GHz homeoffice-communication** scenario is publicly available:

| Source | Link |
|--------|------|
| IEEE DataPort | [Beamformer Ray-Tracing Channel Dataset](https://ieee-dataport.org/documents/beamformer-ray-tracing-channel-dataset) |
| S3 Storage | [homeoffice-communication-28G-raw-data.zip](https://s3-west.nrp-nautilus.io/BeamFormer/dataset/homeoffice-communication-28G-raw-data.zip) |

Download and extract the dataset, then place the SQLite folders at the path set by `sqlite_folder` in [config.m](config.m) (default: `./homeoffice-communication-28G-sqlite/`).

The pipeline expects **ray-tracing output** organized as one SQLite folder per simulation snapshot:

```
<parent_folder>/
├── snapshot_001/
│   ├── tx_v.sqlite    # Vertical polarization paths
│   └── tx_h.sqlite    # Horizontal polarization paths
├── snapshot_002/
│   ├── tx_v.sqlite
│   └── tx_h.sqlite
...
```

Each SQLite database contains the following tables:

| Table | Description |
|-------|-------------|
| `tx` | Transmitter position |
| `rx` | Receiver position(s) and IDs |
| `path` | Path-level data: AoA, AoD, delay, power |
| `interaction` | Reflection/diffraction interaction points |
| `path_utd` | UTD complex E-field coefficients per path per frequency |

---

## Quick Start

**1. Prepare input data**

Place ray-tracing SQLite folders at the path configured by `sqlite_folder` (default: `./homeoffice-communication-28G-sqlite/`).

**2. Configure parameters**

Open [config.m](config.m) and set the key parameters:

```matlab
sqlite_folder       = './homeoffice-communication-28G-sqlite'; % input SQLite path
csi_output_folder   = './csi';  % root output path (subfolders auto-created)
M_tx = 16; N_tx = 16;          % Tx array: 16x16
M_rx = 2;  N_rx = 1;           % Rx array: 2x1
los_enable          = true;     % false = NLOS only dataset
consider_pattern    = false;    % true = apply patch antenna pattern
consider_tilt       = true;     % true = apply beam tilt correction
```

**3. Run the pipeline**

Open MATLAB, navigate to the repository root, and run:

```matlab
main
```

Output `.mat` files will be saved to (suffix reflects active options; empty with default settings):
```
./csi/t16x16_r2x1_csi/
```

---

## Output Format

Each output file `<snapshot_name>-<channel_idx>.mat` contains a single variable:

| Variable | Type | Shape | Description |
|----------|------|-------|-------------|
| `csi` | `complex double` | `[subcarriers × N_rx×M_rx × N_tx×M_tx]` | MIMO CSI tensor |

Default shape: **128 × 2 × 256** (128 subcarriers, 2 Rx elements, 256 Tx elements).

An SQLite metadata file is written alongside the `.mat` files, recording the estimated AoA/AoD angles and beam tilt for each sample.

---

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sqlite_folder` | `'./homeoffice-communication-28G-sqlite'` | Path to the input SQLite folder |
| `csi_output_folder` | `'./csi'` | Root path for all CSI output (subfolders auto-created) |
| `dataset_description` | `'homeoffice-communication'` | Scenario tag (used for radar auto-detection) |
| `freq_zone` | `28` | Frequency band in GHz |
| `M_tx`, `N_tx` | `16, 16` | Tx array rows and columns |
| `M_rx`, `N_rx` | `2, 1` | Rx array rows and columns |
| `path_num` | `25` | Maximum number of paths to retain |
| `los_enable` | `true` | Include LOS paths (`false` = NLOS-only) |
| `consider_pattern` | `false` | Apply patch antenna radiation pattern |
| `consider_tilt` | `false` | Apply beam tilt/steering correction |
| `mask_path` | `false` | Apply random per-path amplitude masking |
| `gpu_mode` | `false` | Offload computation to GPU |

---

## License

This project is released as open-source. See [LICENSE](LICENSE) for details.
