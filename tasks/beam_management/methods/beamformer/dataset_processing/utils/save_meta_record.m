function save_meta_record(v_match_table, tx_about_y, tx_about_z, aod_phi_list, aod_theta_list, polar_component_complex_siso,meta_info_detail_path)


[power,power_ref] = get_path_power(polar_component_complex_siso);
path_num = size(v_match_table, 1);
data_table = table('Size', [path_num, 5], ...
        'VariableTypes', {'double', 'double', 'double', 'double', 'double'}, ...
        'VariableNames', {'path_id', 'phi_rel', 'theta_rel', 'power', 'power_ref'});

for path_idx = 1:path_num
    
    aod_phi_iter = aod_phi_list(path_idx);
    aod_theta_iter = aod_theta_list(path_idx);
    [theta_deg_rel, phi_deg_rel] = get_rel_angle(tx_about_y, tx_about_z, aod_phi_iter, aod_theta_iter);

    data_table.path_id(path_idx) = v_match_table(path_idx,1);
    data_table.phi_rel(path_idx) = phi_deg_rel;
    data_table.theta_rel(path_idx) = theta_deg_rel;
    data_table.power(path_idx) = power(path_idx);
    data_table.power_ref(path_idx) = power_ref;

end

writetable(data_table, meta_info_detail_path, 'Delimiter', ',');
end

function [power,power_ref] = get_path_power(siso_matrix)
siso_matrix_square = abs(siso_matrix).^2;
power_raw = sum(siso_matrix_square, 2);
power_ref = power_raw(1);
power = power_raw / power_ref;
end




function [theta_deg, phi_deg] = get_rel_angle(tx_about_y, tx_about_z, aod_phi, aod_theta)
    %% -------- 基本坐标系变换 --------
    % 1. 将阵列从 YOZ 平面映射到 XOY 平面  
    % 原始阵列平面 (YOZ) → 局部平面 (XOY)
    T_base = [0 1 0;   
               0 0 1;  
               1 0 0];  

    % 2. 旋转阵列
    Ry = rotation_matrix_y(deg2rad(tx_about_y));
    Rz = rotation_matrix_z(deg2rad(tx_about_z));
    R_total = Rz * Ry;

    % 全局 → 局部完整旋转矩阵
    R_global_to_local = T_base * R_total';

    % AOD 射线（直接用全局坐标系绘制）
    theta_rad = deg2rad(aod_theta);
    phi_rad = deg2rad(aod_phi);
    aod_dir_global = [sin(theta_rad)*cos(phi_rad); sin(theta_rad)*sin(phi_rad); cos(theta_rad)];

    % 正确转换 AOD 射线到局部坐标系
    aod_dir_local = R_global_to_local * aod_dir_global;

    [theta_deg, phi_deg] = get_angles_from_vector(aod_dir_local);

    if theta_deg > 90
        theta_deg = 180 - theta_deg;
    end

end

function R = rotation_matrix_y(angle_rad)
    R = [cos(angle_rad), 0, sin(angle_rad);
         0, 1, 0;
        -sin(angle_rad), 0, cos(angle_rad)];
end

function R = rotation_matrix_z(angle_rad)
    R = [cos(angle_rad), -sin(angle_rad), 0;
         sin(angle_rad),  cos(angle_rad), 0;
         0, 0, 1];
end

function [theta_deg, phi_deg] = get_angles_from_vector(v)
    % v: 3x1 向量 [x; y; z]，已在局部坐标系中
    % theta: 仰角 (0° ~ 180°)
    % phi: 方位角 (-180° ~ 180°)

    x = v(1);
    y = v(2);
    z = v(3);

    r = sqrt(x^2 + y^2 + z^2);

    theta_rad = acos(z / r);
    phi_rad = atan2(y, x);

    % 转换成角度
    theta_deg = rad2deg(theta_rad);
    phi_deg = rad2deg(phi_rad);

    % 保持 phi 在 [0, 360)
    if phi_deg < 0
        phi_deg = phi_deg + 360;
    end
end

