function received_complex = take_pattern_influence...
    (polar_component_complex_mimo, aod_phi, aod_theta, aoa_phi,aoa_theta, tilt_info, preloaded_data_tx, preloaded_data_rx)
% aod, aoa should be in rad
[depart_x, depart_y, depart_z] = sph2cart_WI(aod_phi(:),aod_theta(:));
depart_cart = [depart_x, depart_y, depart_z].';% 3,250*36*36
[arrival_x, arrival_y, arrival_z] = sph2cart_WI(aoa_phi(:),aoa_theta(:)); 
arrival_cart = [arrival_x, arrival_y, arrival_z].';% 3,250*36*36

% 原始阵列平面 (YOZ) → 局部平面 (XOY)
T_base = [0 0 1;   % 原 Y → 新 X
           0 1 0;   % 原 Z → 新 Y
           1 0 0];  % 原 X → 新 Z

path_num = size(polar_component_complex_mimo, 1);
freq_num = size(polar_component_complex_mimo, 2);
num_rx = size(polar_component_complex_mimo, 3);
num_tx = size(polar_component_complex_mimo, 4);

tx_rotation_matrix = calculate_rotation_matrix(tilt_info.tx_about_y,tilt_info.tx_about_z);
rx_rotation_matrix = calculate_rotation_matrix(tilt_info.rx_about_y,tilt_info.rx_about_z);

relative_depart_cart = T_base*tx_rotation_matrix'*depart_cart;
relative_arrival_cart = T_base*rx_rotation_matrix'*arrival_cart;
% [phi, theta] = cart2sph_antenna(vec_in_cart)
[relative_depart_sph_phi, relative_depart_sph_theta] = cart2sph_antenna(relative_depart_cart);
[relative_arrival_sph_phi, relative_arrival_sph_theta] = cart2sph_antenna(relative_arrival_cart);

% polar_component_complex_mimo size: path_num*freq*num_rx*num_tx
[gain_theta_aod, phase_theta_aod] = correct_signal_batch(preloaded_data_tx, relative_depart_sph_phi, relative_depart_sph_theta);
gain_theta_aod = reshape(gain_theta_aod, [path_num,1,num_rx,num_tx]);
phase_theta_aod = reshape(phase_theta_aod, [path_num,1,num_rx,num_tx]);
gain_theta_aod = repmat(gain_theta_aod, [1,freq_num,1,1]);
phase_theta_aod = repmat(phase_theta_aod, [1,freq_num,1,1]);

[gain_theta_aoa, phase_theta_aoa] = correct_signal_batch(preloaded_data_rx, relative_arrival_sph_phi, relative_arrival_sph_theta);
gain_theta_aoa = reshape(gain_theta_aoa, [path_num,1,num_rx,num_tx]);
phase_theta_aoa = reshape(phase_theta_aoa, [path_num,1,num_rx,num_tx]);
gain_theta_aoa = repmat(gain_theta_aoa, [1,freq_num,1,1]);
% squeeze(gain_theta_aoa(20,:,1,1)) [-20.1962  -20.1962  -20.1962  -20.1962  -20.1962]
phase_theta_aoa = repmat(phase_theta_aoa, [1,freq_num,1,1]);

coeff_ampt_aoa = 10.^(gain_theta_aoa/20);
delta_phs_aoa = phase_theta_aoa*pi/180;
complex_plus_aoa = coeff_ampt_aoa.*exp(1i*delta_phs_aoa);

coeff_ampt_aod = 10.^(gain_theta_aod/20);
delta_phs_aod = phase_theta_aod*pi/180;
complex_plus_aod = coeff_ampt_aod.*exp(1i*delta_phs_aod);

% received_complex = complex(zeros(size(polar_component_complex_mimo)));
% for path_idx = 1:path_num
%     received_complex(path_idx, :, :, :) = complex_plus_aoa(path_idx, :, :, :).*complex_plus_aod(path_idx, :, :, :).*polar_component_complex_mimo(path_idx, :, :, :);
% end
received_complex = complex_plus_aoa.*complex_plus_aod.*polar_component_complex_mimo;
end

function [gain_theta_output, phase_theta_output] = correct_signal_batch(preloaded_data, aoa_phi, aoa_theta)
    % 获取唯一的theta和phi
    theta_unique = preloaded_data.theta_unique;
    phi_unique = preloaded_data.phi_unique;

    % 获取增益和相位的网格数据
    gain_theta_grid = preloaded_data.gain_theta_grid;
    phase_theta_grid = preloaded_data.phase_theta_grid;

    % 矢量化插值：批量处理aoa_phi和aoa_theta
    % 通过interp2进行批量插值，返回与aoa_phi和aoa_theta相同大小的矩阵
    gain_theta_output = interp2(theta_unique, phi_unique, gain_theta_grid, aoa_theta, aoa_phi, 'linear');
    phase_theta_output = interp2(theta_unique, phi_unique, phase_theta_grid, aoa_theta, aoa_phi, 'linear');
end



function [x_new, y_new, z_new] = sph2cart_WI(aoa_phi,aoa_theta)
    x_new = sin(aoa_theta) .* cos(aoa_phi);
    y_new = sin(aoa_theta) .* sin(aoa_phi);
    z_new = cos(aoa_theta);
end



function [phi, theta] = cart2sph_antenna(vec_in_cart)
    % 输入：3*N 矩阵，表示 N 个点的笛卡尔坐标
    [rows, ~] = size(vec_in_cart);
    if rows ~= 3
        error('输入必须是一个 3xN 矩阵');
    end
    
    % 提取 x, y, z 坐标
    x = vec_in_cart(1, :);
    y = vec_in_cart(2, :);
    z = vec_in_cart(3, :);
    
    % 计算方位角 az
    phi = atan2(y, x); % atan2 返回 [-pi, pi]
    phi = wrapTo2Pi(phi); % [0, 2pi]
    % 计算俯仰角 el
    r = sqrt(x.^2 + y.^2 + z.^2); % 计算到原点的距离
    theta = acos(z ./ r);
    
    % 将 az 和 el 转换为度数 1*N
    phi = rad2deg(phi).';
    theta = rad2deg(theta).';
    
end 

% function values = get_values_from_indices(A, idx)
%     % get_values_from_indices 从三维矩阵 A 中通过索引矩阵 idx 提取值
%     % 输入:
%     %   A   - 三维矩阵
%     %   idx - N×3 的索引矩阵，每行表示 [i, j, k] 的三维索引
%     % 输出:
%     %   values - 从 A 中提取的对应于 idx 中每个索引的值
% 
%     % 获取三维矩阵的尺寸
%     [X, Y, Z] = size(A);
%     % 检查 idx 是否为 N×3 的矩阵
%     if size(idx, 2) ~= 3
%         error('索引矩阵必须是 N×3 的矩阵');
%     end
% 
%     % 将 N×3 的索引矩阵转换为线性索引
%     linear_indices = sub2ind([X, Y, Z], idx(:,1), idx(:,2), idx(:,3));
% 
%     % 获取对应的值
%     values = A(linear_indices);
% end
