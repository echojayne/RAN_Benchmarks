function polar_component_complex_siso = synthesis_elec_data(v_elec, h_elec,aod_phi, aod_theta, aoa_phi, aoa_theta,tilt_info,freq_list)
    beta_WI = 1.00039160809;

    tx_rotate_z = tilt_info.tx_about_z;
    tx_rotate_y = tilt_info.tx_about_y;
    rx_rotate_z = tilt_info.rx_about_z;
    rx_rotate_y = tilt_info.rx_about_y;
    
    aod_phi_siso = squeeze(aod_phi(:,1,1));
    aod_theta_siso = squeeze(aod_theta(:,1,1));
    aoa_phi_siso = squeeze(aoa_phi(:,1,1));
    aoa_theta_siso = squeeze(aoa_theta(:,1,1));
    path_num = length(aod_phi_siso);
    freq_num = length(squeeze(v_elec(1,:,1)));
    freq_list = reshape(freq_list,[1,freq_num]);
    ampt_coeff = physconst('LightSpeed')*sqrt(beta_WI/(8*pi*377))./freq_list;
    polar_component_complex_siso = complex(zeros(path_num,freq_num));

    for path_idx = 1:path_num
        [Tx_H_coeff, Tx_V_coeff] = calculateTiltCoeff...
            (tx_rotate_y, tx_rotate_z, aod_phi_siso(path_idx), aod_theta_siso(path_idx));
        [Rx_H_coeff, Rx_V_coeff] = calculateTiltCoeff...
            (rx_rotate_y, rx_rotate_z, aoa_phi_siso(path_idx), aoa_theta_siso(path_idx));
        % elec_syn: 384*4
        elec_syn = Tx_H_coeff*h_elec(path_idx, :, 3:6) + Tx_V_coeff*v_elec(path_idx, :, 3:6);
        elec_syn = squeeze(elec_syn);
        % Rx_V_coeff = 1
        project_vector = Rx_V_coeff*elec_syn(:,1:2) + Rx_H_coeff*elec_syn(:,3:4);
        polar_component_complex_siso(path_idx, :) = ampt_coeff.*(project_vector(:,1) + 1i*project_vector(:,2)).';
    end
    
end



% **********************************************************************
function [Rx_H_coeff, Rx_V_coeff] = calculateTiltCoeff(theta_y, phi_z, phi, theta)
    get_vertical_component = @(a,b) (a - dot(a,b)*b/norm(b)/norm(b))/norm((a - dot(a,b)*b/norm(b)/norm(b)));
    get_cos_value = @(a,b) dot(a,b)/norm(a)/norm(b);
    NewRx = rotate_z_axis(theta_y, phi_z);
    [x_arrival, y_arrival, z_arrival] = sph2cart_WI(phi,theta);
    [y_axis, z_axis] = calculate_yz_axes(phi, theta);
    vertical_component = get_vertical_component(NewRx, [x_arrival, y_arrival, z_arrival]');
    Rx_H_coeff = get_cos_value(vertical_component, y_axis);
    Rx_V_coeff = get_cos_value(vertical_component, z_axis);
end

function [phi_axis, theta_axis] = calculate_yz_axes(aoa_phi, aoa_theta)
    theta_axis = [cos(aoa_phi)*cos(aoa_theta); sin(aoa_phi)*cos(aoa_theta); -sin(aoa_theta)];
    phi_axis = [-sin(aoa_phi); cos(aoa_phi); 0];
end

function z_new = rotate_z_axis(theta_y, phi_z)
    % y 轴旋转矩阵
    R_y = [cos(theta_y), 0, sin(theta_y);
        0, 1, 0;
        -sin(theta_y), 0, cos(theta_y)];
    % z 轴旋转矩阵
    R_z = [cos(phi_z), -sin(phi_z), 0;
        sin(phi_z), cos(phi_z), 0;
        0, 0, 1];
    % 初始 z 轴向量
    z_old = [0; 0; -1];
    % 先沿 y 轴旋转，再沿 z 轴旋转
    z_new = R_z * R_y * z_old;
end


function [x_new, y_new, z_new] = sph2cart_WI(aoa_phi,aoa_theta)
    x_new = sin(aoa_theta) .* cos(aoa_phi);
    y_new = sin(aoa_theta) .* sin(aoa_phi);
    z_new = cos(aoa_theta);
end