function mimo_match_struct = fetch_mimo_data_for_match_table(sqlite_struct, match_table, tx_array,rx_array,freq_list)
    num_path = length(match_table);
    num_freq = length(freq_list);
    num_tx = size(tx_array,1);
    num_rx = size(rx_array,1);

    mimo_complex = complex(zeros(num_path, num_freq,num_rx,num_tx));
    mimo_aod_phi = zeros(num_path,num_rx,num_tx);
    mimo_aod_theta = zeros(num_path,num_rx,num_tx);
    mimo_aoa_phi = zeros(num_path,num_rx,num_tx);
    mimo_aoa_theta = zeros(num_path,num_rx,num_tx);
    
    for path_iter_idx = 1:num_path
        path_iter = match_table(path_iter_idx);
        interaction_filter = (sqlite_struct.interaction.path_id == path_iter);
        select_data = sqlite_struct.interaction(interaction_filter,:);
        interaction_by_path = [cast(select_data.path_id,'double'),...
                               cast(select_data.interaction_type_id, 'double'),...
                               select_data.x, select_data.y, select_data.z];
        if(path_iter == 0)
            continue
        end
        lambda_list = physconst('LightSpeed')./freq_list;
        [mimo_complex_1path, mimo_aod_phi_1path, mimo_aod_theta_1path, mimo_aoa_phi_1path, mimo_aoa_theta_1path] = ...
            construct_mimo_info(tx_array,rx_array, lambda_list,interaction_by_path);
        mimo_complex(path_iter_idx, :, :, :) = mimo_complex_1path;
        mimo_aod_phi(path_iter_idx, :, :) = mimo_aod_phi_1path;
        mimo_aod_theta(path_iter_idx, :, :) = mimo_aod_theta_1path;
        mimo_aoa_phi(path_iter_idx, :,:) = mimo_aoa_phi_1path;
        mimo_aoa_theta(path_iter_idx, :,:) = mimo_aoa_theta_1path;
    end
    mimo_match_struct.mimo_complex = mimo_complex;
    mimo_match_struct.mimo_aod_phi = mimo_aod_phi*pi/180;
    mimo_match_struct.mimo_aod_theta = mimo_aod_theta*pi/180;
    mimo_match_struct.mimo_aoa_phi = mimo_aoa_phi*pi/180;
    mimo_match_struct.mimo_aoa_theta = mimo_aoa_theta*pi/180;
end

function [mimo_complex_1path, mimo_aod_phi_1path, mimo_aod_theta_1path, mimo_aoa_phi_1path, mimo_aoa_theta_1path] = ...
    construct_mimo_info(tx_array,rx_array, subcarries_lambda_list,interaction_by_path)
% tx1rx1_phs: antenna pairs(tx1 - rx1) phase [-pi, pi]
% interaction_by_path: the interaction of single path and single frequency
%                     [path_id,interaction_type_id,x,y,z]
% output: phi, theta - degree
    
    tx_ref_index = 1;
    rx_ref_index = 1;
    tx_interaction = [0, 0, tx_array(1,:)]; % the first column seems to be useless
    rx_interaction = [0, 1, rx_array(1,:)];
    interaction_by_path = [tx_interaction;interaction_by_path;rx_interaction];

    num_tx = size(tx_array,1);
    num_rx = size(rx_array,1);
    mimo_aod_phi_1path = zeros(num_rx,num_tx);
    mimo_aod_theta_1path = zeros(num_rx,num_tx);
    mimo_aoa_phi_1path = zeros(num_rx,num_tx);
    mimo_aoa_theta_1path = zeros(num_rx,num_tx);
    inte_len_by_path = size(interaction_by_path,1);
    inte_type_by_path = interaction_by_path(:,2);
    if(any(inte_type_by_path == 3))
        new_source_tx_idx = find(inte_type_by_path == 3, 1, 'last');
        eff_tx = interaction_by_path(new_source_tx_idx,3:5);
        for inter_id_ii = new_source_tx_idx+1:inte_len_by_path-1
            if(inte_type_by_path(inter_id_ii)==4)
                continue;
            end
            eff_tx = mirroring_effect(eff_tx, interaction_by_path(inter_id_ii-1,3:5), ...
                interaction_by_path(inter_id_ii,3:5),interaction_by_path(inter_id_ii+1,3:5));
        end
        tx2rx_dist = vecnorm(eff_tx-rx_array,2,2);
        tx2rx_dist_delta = tx2rx_dist - tx2rx_dist(tx_ref_index);

        new_source_rx_idx = find(inte_type_by_path == 3, 1, 'first');
        eff_rx = interaction_by_path(new_source_rx_idx,3:5);
        for inter_id_ii = new_source_rx_idx-1:-1:2
            if(inte_type_by_path(inter_id_ii)==4)
                continue;
            end
            eff_rx = mirroring_effect(eff_rx, interaction_by_path(inter_id_ii+1,3:5), ...
                interaction_by_path(inter_id_ii,3:5),interaction_by_path(inter_id_ii-1,3:5));
        end
        rx2tx_dist = vecnorm(eff_rx-tx_array,2,2);
        rx2tx_dist_delta = rx2tx_dist - rx2tx_dist(rx_ref_index);
        tx_rx_dist_delta = tx2rx_dist_delta + rx2tx_dist_delta.';
        for rx_i = 1:num_rx
            for tx_i = 1:num_tx
                [mimo_aod_theta_1path(rx_i,tx_i), mimo_aod_phi_1path(rx_i,tx_i)]...
                    = calculateAngles(tx_array(tx_i,:),eff_rx);
                [mimo_aoa_theta_1path(rx_i,tx_i), mimo_aoa_phi_1path(rx_i,tx_i)]...
                    = calculateAngles(rx_array(rx_i,:),eff_tx);
            end
        end        
    else
        new_source_tx_idx = 1;
        num_eff_tx = num_tx;
        eff_tx = tx_array;

        for inter_id_ii = new_source_tx_idx+1:inte_len_by_path-1
            if(inte_type_by_path(inter_id_ii)==4)
                continue;
            end
            eff_tx = mirroring_effect(eff_tx, interaction_by_path(inter_id_ii-1,3:5), ...
                interaction_by_path(inter_id_ii,3:5),interaction_by_path(inter_id_ii+1,3:5));
        end

        new_source_rx_idx = inte_len_by_path;
        eff_rx = rx_array;
        for inter_id_ii = new_source_rx_idx-1:-1:2
            if(inte_type_by_path(inter_id_ii)==4)
                continue;
            end
            eff_rx = mirroring_effect(eff_rx, interaction_by_path(inter_id_ii+1,3:5), ...
                interaction_by_path(inter_id_ii,3:5),interaction_by_path(inter_id_ii-1,3:5));
        end

        tx_rx_dist = zeros(num_rx, num_eff_tx);
        for tx_i = 1:num_eff_tx
            tx_rx_dist(:,tx_i) = vecnorm(eff_tx(tx_i,:) - rx_array, 2, 2);
        end
        tx_rx_dist_delta = tx_rx_dist - tx_rx_dist(rx_ref_index,tx_ref_index);
        for rx_i = 1:num_rx
            for tx_i = 1:num_tx
                [mimo_aod_theta_1path(rx_i,tx_i), mimo_aod_phi_1path(rx_i,tx_i)]...
                    = calculateAngles(tx_array(tx_i,:),eff_rx(rx_i,:));
                [mimo_aoa_theta_1path(rx_i,tx_i), mimo_aoa_phi_1path(rx_i,tx_i)]...
                    = calculateAngles(rx_array(rx_i,:),eff_tx(tx_i,:));
            end
        end     
        
    end
    
    num_subcarriers = length(subcarries_lambda_list);
    tx_rx_dist_delta_up_dim = reshape(tx_rx_dist_delta, [1, num_rx, num_tx]);
    tx_rx_dist_delta_all_freq = repmat(tx_rx_dist_delta_up_dim,[num_subcarriers,1,1]);
    subcarries_lambda_list_up_dim = reshape(subcarries_lambda_list, [num_subcarriers,1,1]);
    subcarries_lambda_list_broadcast = repmat(subcarries_lambda_list_up_dim, [1, num_rx, num_tx]);
    compensation_all_freqs = -2*pi*tx_rx_dist_delta_all_freq./subcarries_lambda_list_broadcast;
    mimo_complex_1path = exp(1i*compensation_all_freqs);
end

function px_image = mirroring_effect(px, pt, pi, pr)
    eti = (pi-pt)/norm(pi-pt);
    eir = (pr-pi)/norm(pr-pi);
    en = (eti-eir)/norm(eti-eir);
    px_image = px - ((px-pi)*en.')*en*2;
end

function [theta_deg, phi_deg] = calculateAngles(A, B)
    % 计算从点A看点B的俯仰角和方位角
    directionVector = B - A;
    
    % 提取方向向量的坐标
    x = directionVector(1);
    y = directionVector(2);
    z = directionVector(3);
    
    % 计算距离
    distance = sqrt(x^2 + y^2 + z^2);
    
    % 计算俯仰角 (θ)
    theta = acos(z / distance);
    
    % 计算方位角 (φ)
    phi = atan2(y, x);
    
    % 将结果转换为度
    theta_deg = rad2deg(theta);
    phi_deg = rad2deg(phi);
end