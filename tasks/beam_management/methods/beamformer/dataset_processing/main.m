clc,clear;
addpath(genpath(pwd));
config;
path_list = list_paths(parent_folder_path, pod_id, total_server_num);
total_steps = length(path_list);
for path_list_idx = 1:length(path_list)
    
    sqlite_folder_path = path_list{path_list_idx};
    [~, folder_name] = fileparts(sqlite_folder_path);

    % read sqlite data
    sqlite_struct = read_sqlite(sqlite_folder_path,  splite_num);

    if radar_setting == true
        % only 1 tx, 1 rx, so channel num = 1
        if all(sqlite_struct.path_v.departure_theta > 150)
            fprintf('Dirty sqlite: %s\n', sqlite_folder_path);
            continue;
        else
            sqlite_struct = filter_dirty_paths(sqlite_struct);
        end
    end

    if sqlite_struct.is_valid == false
        fprintf('Invalid sqlite data in %s\n', sqlite_folder_path);
        continue;
    end
    [channel_num, channel_list, angle_per_channel] = get_valid_channels(sqlite_struct.path_v);


    % generate tilt angles
    tilt_info_array = generate_tilt_info(channel_num, channel_list, angle_per_channel, max_tolerance_angle_deg, consider_tilt);

    for channel_id_idx = 1:channel_num
        
        channel_id = channel_list(channel_id_idx);

        tilt_info = tilt_info_array(channel_id);

        % generate tx,rx mimo arrays, tilt_info(rad)
        [tx_array, rx_array] = generate_array(sqlite_struct.tx, sqlite_struct.rx(sqlite_struct.rx.rx_id == channel_id-1,:), ...
            tx_elements_setting, rx_elements_setting, tilt_info);

        % create matched tables
        [v_match_table, h_match_table] = generate_match_tables(sqlite_struct, channel_id);

        % remove LOS paths if los_enable is false
        % LOS paths have no entries in the interaction table
        if ~los_enable
            nlos_path_ids = unique(sqlite_struct.interaction.path_id);
            los_mask = ismember(v_match_table, nlos_path_ids);
            v_match_table = v_match_table(los_mask);
            h_match_table = h_match_table(los_mask);
            if isempty(v_match_table)
                continue;
            end
        end

        % fetch elec data for each matched table 250*384*6
        v_elec_match_data = fetch_elec_data_for_match_table(sqlite_struct.path_utd_v, v_match_table, subcarriers_num); % √-nontilt
        h_elec_match_data = fetch_elec_data_for_match_table(sqlite_struct.path_utd_h, h_match_table, subcarriers_num);

        
        % fetch mimo data for each matched table,
        % mimo_match_struct: {aoa(rad), aod(rad), phs-compensation-complex}, size: path_num*Rx*Tx
        v_mimo_match_struct = fetch_mimo_data_for_match_table(sqlite_struct, v_match_table,tx_array,rx_array,freq_list);% √-nontilt
        if check_estimate_angles(v_mimo_match_struct, sqlite_struct, channel_id, v_match_table)
            continue;
        end        
        
        % generate polar-component complex, size: path_num*freq
        polar_component_complex_siso = synthesis_elec_data(v_elec_match_data, h_elec_match_data, v_mimo_match_struct.mimo_aod_phi, ...
            v_mimo_match_struct.mimo_aod_theta, v_mimo_match_struct.mimo_aoa_phi,v_mimo_match_struct.mimo_aoa_theta, tilt_info,freq_list); % √-nontilt

        if mask_path
            path_num = size(polar_component_complex_siso, 1);
            path_weight_vec = 0.1 + (1.0 - 0.1) * rand(path_num, 1);  % [path_num x 1]
            polar_component_complex_siso = polar_component_complex_siso .* path_weight_vec;  
        end
        if gpu_mode==true
            polar_component_complex_siso = gpuArray(polar_component_complex_siso);
            v_mimo_match_struct = convertStructToGpu(v_mimo_match_struct);
            preloaded_data = convertStructToGpu(preloaded_data);
            tilt_info = convertStructToGpu(tilt_info);
        end
        
        % convert siso complex to mimo complex, size: path_num*freq*num_rx*num_tx
        polar_component_complex_mimo = siso2mimo(polar_component_complex_siso, v_mimo_match_struct.mimo_complex);
        % pattern influence
        if consider_pattern
            received_complex = take_pattern_influence(polar_component_complex_mimo, v_mimo_match_struct.mimo_aod_phi, ...
                v_mimo_match_struct.mimo_aod_theta, v_mimo_match_struct.mimo_aoa_phi,v_mimo_match_struct.mimo_aoa_theta, tilt_info, preloaded_data_tx, preloaded_data_rx);
        else
            received_complex = polar_component_complex_mimo;
        end
        % convert received_complex to csi
        csi = squeeze(sum(received_complex, 1));
        if gpu_mode==true
            csi = gather(csi);
        end
        % save csi
        csi_file_path = fullfile(csi_folder_path,sprintf('%s-%d.mat', folder_name,channel_id_idx));
        meta_info_detail_path = fullfile(meta_info_detail_folder,sprintf('%s-%d.csv', folder_name,channel_id_idx));
        save(csi_file_path, 'csi');
        write_csi_angle_meta_info(sqlite_conn, csi_file_path, angle_per_channel(channel_id), tilt_info)
        save_meta_record(v_match_table, rad2deg(tilt_info.tx_about_y), rad2deg(tilt_info.tx_about_z), rad2deg(v_mimo_match_struct.mimo_aod_phi(:,1,1)), rad2deg(v_mimo_match_struct.mimo_aod_theta(:,1,1)), polar_component_complex_siso,meta_info_detail_path);
        % clear received_complex csi polar_component_complex_mimo v_mimo_match_struct
    end
    msg = sprintf('Processing %d/%d...', path_list_idx, total_steps);
    progressBar(path_list_idx, total_steps, msg);
end
close(sqlite_conn);
