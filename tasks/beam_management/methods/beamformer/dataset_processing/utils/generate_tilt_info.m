function tilt_info_array = generate_tilt_info(channel_num, channel_list, angle_per_channel, max_tolerance_angle_deg, consider_tilt)
    max_channel_id = max(channel_list);
    tilt_info_array(1, max_channel_id) = struct();
    for ii = 1:channel_num
        channel_id = channel_list(ii);
        aod_phi = angle_per_channel(channel_id).aod_phi;
        aod_theta = angle_per_channel(channel_id).aod_theta;
        [tx_phi_ori, tx_theta_ori] = generate_phi_theta_within_angle(aod_phi, aod_theta, max_tolerance_angle_deg);
        aoa_phi = angle_per_channel(channel_id).aoa_phi;
        aoa_theta = angle_per_channel(channel_id).aoa_theta;
        [rx_phi_ori, rx_theta_ori] = generate_phi_theta_within_angle(aoa_phi, aoa_theta, max_tolerance_angle_deg);
        tilt_info_array(channel_id).tx_about_y = (tx_theta_ori - 90)*pi/180;
        tilt_info_array(channel_id).tx_about_z = tx_phi_ori*pi/180;
        tilt_info_array(channel_id).rx_about_y = (rx_theta_ori - 90)*pi/180;
        tilt_info_array(channel_id).rx_about_z = rx_phi_ori*pi/180; 
        if consider_tilt == false
            tilt_info_array(channel_id).tx_about_y = 0;
            tilt_info_array(channel_id).tx_about_z = 0;
            tilt_info_array(channel_id).rx_about_y = 0;
            tilt_info_array(channel_id).rx_about_z = 0; 
        end 
    end
end


function [phi_new, theta_new] = generate_phi_theta_within_angle(phi_deg, theta_deg, max_angle_deg)
% 输入：
%   phi_deg        - 原始方位角 (度)
%   theta_deg      - 原始仰角 (度)
%   max_angle_deg  - 最大允许夹角 (度)
%
% 输出：
%   phi_new        - 新方位角 (度)
%   theta_new      - 新仰角 (度)

% 转为弧度
phi = deg2rad(phi_deg);
theta = deg2rad(theta_deg);

% 原方向单位向量
x = sin(theta) * cos(phi);
y = sin(theta) * sin(phi);
z = cos(theta);
center_vec = [x; y; z];

cos_max_angle = cosd(max_angle_deg);

% 采样直到满足夹角条件
while true
    % 随机单位向量（均匀分布于球面）
    vec = randn(3, 1);
    vec = vec / norm(vec);

    % 判断夹角是否小于阈值
    if dot(vec, center_vec) >= cos_max_angle
        break;
    end
end

% 转为球坐标
x_new = vec(1);
y_new = vec(2);
z_new = vec(3);
r = norm(vec);
theta_new = acos(z_new / r);
phi_new = atan2(y_new, x_new);

% 转为角度并归一化
theta_new = rad2deg(theta_new);
phi_new = mod(rad2deg(phi_new), 360);
end
