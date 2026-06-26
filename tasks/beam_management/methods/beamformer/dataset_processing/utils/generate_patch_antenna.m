function preloaded_data = generate_patch_antenna(phi_0, theta_0, G_max, h_bw, v_bw, output_file, save_uan)
%GENERATE_PATCH_UAN  生成Patch天线方向图文件(.uan)、绘制3D方向图，并返回预加载数据
%   phi_0, theta_0: 主瓣指向的方位角和仰角（度）
%   G_max: 最大方向增益（dBi）
%   h_bw, v_bw: 水平和垂直方向的3dB波束宽度（度）
%   output_file: 输出文件路径和名称（字符串）
%   save_uan: 是否保存uan文件 (true/false)
%
%   返回:
%   preloaded_data: 预处理后的方向图数据

% 角度标准化
phi_0 = mod(phi_0, 360);
theta_0 = max(0, min(theta_0, 180));

% 角度采样设置
phi_vals = 0:1:360;
theta_vals = 0:1:180;

numTheta = length(theta_vals);
numPhi = length(phi_vals);

G_theta_matrix = zeros(numTheta, numPhi);
G_phi_matrix = zeros(numTheta, numPhi);

if save_uan
    fid = fopen(output_file, 'w');
    if fid == -1
        error('无法创建输出文件：%s', output_file);
    end
    % 写入文件头
    fprintf(fid, 'begin_<parameters>\n');
    fprintf(fid, 'format free\n');
    fprintf(fid, 'phi_min 0\nphi_max 360\nphi_inc 1\n');
    fprintf(fid, 'theta_min 0\ntheta_max 180\ntheta_inc 1\n');
    fprintf(fid, 'complex\nmag_phase\npattern gain\nmagnitude dB\n');
    fprintf(fid, 'direction degrees\nphase degrees\npolarization theta_phi\n');
    fprintf(fid, 'NetInputPower 1.0\n');
    fprintf(fid, 'end_<parameters>\n');
end

% 计算增益
for it = 1:numTheta
    theta = theta_vals(it);
    for ip = 1:numPhi
        phi = phi_vals(ip);
        dPhi = mod(abs(phi - phi_0), 360);
        dPhi = min(dPhi, 360 - dPhi);
        dTheta = abs(theta - theta_0);

        if sind(theta_0) < 1e-12
            dPhi_eff = 0;
        else
            dPhi_eff = dPhi * sind(theta_0);
        end

        F_h = max(0, cosd((dPhi_eff / (h_bw / 2)) * 45));
        F_v = max(0, cosd((dTheta / (v_bw / 2)) * 45));

        if F_h == 0 || F_v == 0
            G_theta_dB = G_max - 40;
        else
            G_theta_dB = G_max + 10 * log10((F_h * F_v)^2);
        end
        G_phi_dB = G_theta_dB;

        G_theta_matrix(it, ip) = G_theta_dB;
        G_phi_matrix(it, ip) = G_phi_dB;

        if save_uan
            fprintf(fid, '%.1f %.1f %.3f %.3f 0 0\n', theta, phi, G_theta_dB, G_phi_dB);
        end
    end
end

if save_uan
    fclose(fid);
    fprintf('方向图文件已生成: %s\n', output_file);

    
    % 可视化
    G_linear = 10.^(G_theta_matrix / 10);
    G_linear_norm = G_linear / max(G_linear(:));
    
    [Phi_grid, Theta_grid] = meshgrid(phi_vals, theta_vals);
    Phi_rad = deg2rad(Phi_grid);
    Theta_rad = deg2rad(Theta_grid);
    
    R = sqrt(G_linear_norm);
    X = R .* sin(Theta_rad) .* cos(Phi_rad);
    Y = R .* sin(Theta_rad) .* sin(Phi_rad);
    Z = R .* cos(Theta_rad);
    
    figure;
    surf(X, Y, Z, G_theta_matrix, 'EdgeColor', 'none');
    colormap(jet); colorbar;
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('Patch天线方向图 (主瓣 \\phi_0=%.1f°, \\theta_0=%.1f°, h_{bw}=%.1f°, v_{bw}=%.1f°)', ...
        phi_0, theta_0, h_bw, v_bw));
    axis equal;
    grid on;
    view(45, 30);
end

% 如果不保存文件，直接使用当前数据生成
preloaded_data.theta_unique = theta_vals';
preloaded_data.phi_unique = phi_vals';
preloaded_data.gain_theta_grid = G_theta_matrix';
preloaded_data.phase_theta_grid = zeros(size(G_theta_matrix'));

end
