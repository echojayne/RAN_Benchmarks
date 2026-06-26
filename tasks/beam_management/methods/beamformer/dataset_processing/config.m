% =========================================================================
% USER PARAMETERS — edit this section before running
% =========================================================================

% Scenario name: used to locate the input SQLite folder and name the output.
% The input folder is: ../sqlite-from-rclone/<dataset_description>-<freq_zone>G-sqlite/
dataset_description = 'homeoffice-communication';

% Carrier frequency band in GHz. Currently only 28 GHz is supported.
freq_zone = 28;

% Path to the input SQLite folder.
sqlite_folder = './homeoffice-communication-28G-sqlite';

% Root path for all CSI output (subfolders are created automatically).
csi_output_folder = './csi';

% Tx antenna array dimensions (rows x columns).
M_tx = 16;
N_tx = 16;

% Rx antenna array dimensions (rows x columns).
M_rx = 2;
N_rx = 1;

% Set to false to exclude LOS paths, producing an NLOS-only dataset.
los_enable = true;

% Set to true to apply the patch antenna radiation pattern to the CSI.
consider_pattern = false;

% Set to true to apply beam tilt / steering correction based on AoA/AoD.
consider_tilt = true;

% Set to true to apply random per-path amplitude masking (data augmentation).
mask_path = false;

% Set to true to offload computation to GPU (requires Parallel Computing Toolbox).
gpu_mode = false;

% Maximum number of ray-tracing paths to retain per channel sample.
path_num = 25;

% =========================================================================
% INTERNAL SETTINGS — no need to edit below
% =========================================================================

c0 = physconst('LightSpeed');

function elements_setting = init_elements_setting(num_y, num_z, spacing_y, spacing_z)
    elements_setting.num_y = num_y;
    elements_setting.num_z = num_z;
    elements_setting.num_total = num_y * num_z;
    elements_setting.spacing_y = spacing_y;
    elements_setting.spacing_z = spacing_z;
end

% Radar mode is auto-detected from the scenario name.
radar_setting = false;
if contains(dataset_description, 'radar')
    radar_setting = true;
end

% Build output folder suffix from enabled options.
csi_folder_suffix = '';
if consider_pattern
    csi_folder_suffix = [csi_folder_suffix, '-patch'];
end
if ~consider_tilt
    csi_folder_suffix = [csi_folder_suffix, '-no-rotate'];
end
if mask_path
    csi_folder_suffix = [csi_folder_suffix, '-mask-path'];
end
if ~los_enable
    csi_folder_suffix = [csi_folder_suffix, '-nlos'];
end

% Input and output folder paths.
parent_folder_path = sqlite_folder;
csi_folder = [csi_output_folder, csi_folder_suffix];

% Frequency band parameters (28 GHz, 128 subcarriers, 240 kHz spacing).
if freq_zone == 28
    spacing = 240e3;
    mid_freq = 27.925e9;
    start_freq = mid_freq - 64*spacing;
    end_freq   = mid_freq + 63*spacing;
else
    disp('60GHz is not supported yet');
end

lambda_mid_freq = c0 / mid_freq;
subcarriers_num = 128;
antenna_spacing_per_lambda = 0.5;

tx_elements_setting = init_elements_setting(N_tx, M_tx, antenna_spacing_per_lambda*lambda_mid_freq, antenna_spacing_per_lambda*lambda_mid_freq);
rx_elements_setting = init_elements_setting(N_rx, M_rx, antenna_spacing_per_lambda*lambda_mid_freq, antenna_spacing_per_lambda*lambda_mid_freq);

splite_num = 2;    % number of segments for chunked SQLite reading

freq_list = linspace(start_freq, end_freq, subcarriers_num);
wave_len  = @(freq) physconst('LightSpeed') / freq;
uan_path  = fullfile(pwd, "utils", "half-wave-dipole.uan");

% Maximum steering angle tolerance for beam tilt correction (degrees).
max_tolerance_angle_deg = 75;
if radar_setting
    max_tolerance_angle_deg = 60;
end

% Patch antenna parameters (used when consider_pattern = true).
bw    = 60;
h_bw  = bw;
v_bw  = bw;
G_max = 10;
preloaded_data_tx = generate_patch_antenna(0, 0, G_max, h_bw, v_bw, 'utils/patch.uan', false);
preloaded_data_rx = generate_patch_antenna(0, 0, G_max, h_bw, v_bw, 'utils/patch.uan', false);

% Worker ID and random seed (single-machine: pod_id = 1).
pod_id = 1;
rng(round(pod_id));
total_server_num = 1;

% Output folder paths.
csi_folder_path         = fullfile(csi_folder, sprintf('t%dx%d_r%dx%d_csi',            M_tx, N_tx, M_rx, N_rx));
meta_info_folder        = fullfile(csi_folder, sprintf('t%dx%d_r%dx%d_metainfo',        M_tx, N_tx, M_rx, N_rx));
meta_info_detail_folder = fullfile(csi_folder, sprintf('t%dx%d_r%dx%d_metainfo_detail', M_tx, N_tx, M_rx, N_rx));
if ~exist(csi_folder_path, 'dir')
    mkdir(csi_folder_path);
end
if ~exist(meta_info_folder, 'dir')
    mkdir(meta_info_folder);
end
if ~exist(meta_info_detail_folder, 'dir')
    mkdir(meta_info_detail_folder);
end

% Create (or reset) the per-run metadata SQLite file.
meta_info_sqlite_path = fullfile(meta_info_folder, [num2str(pod_id), '.sqlite']);
if exist(meta_info_sqlite_path, 'file')
    delete(meta_info_sqlite_path);
end
sqlite_conn = sqlite(meta_info_sqlite_path, 'create');
