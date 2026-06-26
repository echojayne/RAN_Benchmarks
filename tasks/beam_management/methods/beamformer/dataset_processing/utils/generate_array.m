function [tx_array, rx_array] = generate_array(tx_table, rx_table, tx_elements_setting, rx_elements_setting, tilt_info)

tx_array_wo_origin = generateAntennaArray_YOZ(tilt_info.tx_about_z, tilt_info.tx_about_y, tx_elements_setting.num_y, tx_elements_setting.spacing_y, tx_elements_setting.num_z, tx_elements_setting.spacing_z);

tx_array = tx_array_wo_origin + [tx_table.x, tx_table.y, tx_table.z];

rx_array_wo_origin = generateAntennaArray_YOZ(tilt_info.rx_about_z, tilt_info.rx_about_y, rx_elements_setting.num_y, rx_elements_setting.spacing_y, rx_elements_setting.num_z, rx_elements_setting.spacing_z);

rx_array = rx_array_wo_origin + [rx_table.x, rx_table.y, rx_table.z];

end
function coordinates = generateAntennaArray_YOZ(h_angle, v_angle, y_num, y_spacing, z_num, z_spacing)
    % 在 y-z 平面上生成天线阵列坐标，并应用旋转
    % 输入参数 y_num/z_num 分别为 y 和 z 方向上的天线数
    % y_spacing/z_spacing 为对应方向间距
    
    % 创建基础阵列平面, 从 (0, 0) 开始
    [y_idx, z_idx] = meshgrid(0:(y_num-1), 0:(z_num-1));
    
    % 计算在 y 和 z 方向上的坐标（此时是列方向为 y，行方向为 z）
    y_idx = y_idx * y_spacing;
    z_idx = z_idx * z_spacing;

    % 平移到中心
    y_center = (y_num - 1) * y_spacing / 2;
    z_center = (z_num - 1) * z_spacing / 2;
    
    y_idx = y_idx - y_center;
    z_idx = z_idx - z_center;

    % x方向坐标恒为 0（位于 y-z 平面）
    x_idx = zeros(size(y_idx));
    
    % 合成坐标，注意顺序是 (x, y, z)
    coordinates = [x_idx(:), y_idx(:), z_idx(:)]; % 注意，这里的矩阵和实际阵列摆放关于水平方向对称！

    % 构造旋转矩阵
    Rz = [cos(h_angle), -sin(h_angle), 0;
          sin(h_angle),  cos(h_angle), 0;
          0,             0,            1];
    
    Ry = [cos(v_angle),  0, sin(v_angle);
          0,             1, 0;
         -sin(v_angle),  0, cos(v_angle)];
    
    R = Rz * Ry;
    
    % 旋转坐标
    coordinates = (R * coordinates')';
end


% function coordinates = generateAntennaArray(h_angle, v_angle, x_num, x_spacing, y_num, y_spacing)
%     % 生成天线阵列的坐标, 默认z轴无elements分布
%     % 输入:
%     %   h_angle   - 水平方向倾角 (rad)
%     %   v_angle   - 垂直方向倾角 (rad)
%     %   x_num     - x方向的天线元数量, col's num, N
%     %   x_spacing - x方向的天线元间隔
%     %   y_num     - y方向的天线元数量, row's num, M
%     %   y_spacing - y方向的天线元间隔
%     % 输出:
%     %   coordinates - 每个天线的三维坐标 (x, y, z) 列表
% 
%     h_angle_rad = h_angle;
%     v_angle_rad = v_angle;
% 
%     % 创建基础阵列平面, 天线坐标从 (0, 0) 开始
%     [x_idx, y_idx] = meshgrid(0:(x_num-1), 0:(y_num-1));
% 
%     % 计算阵列在 x 和 y 方向上的间隔
%     x_idx = x_idx * x_spacing;
%     y_idx = y_idx * y_spacing;
% 
%     % 将阵列中心平移至原点
%     x_center = (x_num - 1) * x_spacing / 2;
%     y_center = (y_num - 1) * y_spacing / 2;
% 
%     x_idx = x_idx - x_center;  % 平移x方向坐标
%     y_idx = y_idx - y_center;  % 平移y方向坐标
% 
%     % 初始的平面坐标 (x, y, 0)，z=0代表平面上不考虑z方向分布
%     z_idx = zeros(size(x_idx));
% 
%     % 将 (x, y, z) 坐标组合成一个三维坐标数组
%     coordinates = [x_idx(:), y_idx(:), z_idx(:)];
% 
%     % 构造旋转矩阵 (绕z轴旋转水平角，绕y轴旋转垂直角)
%     % 水平倾角 h_angle -> 绕 z 轴旋转
%     Rz = [cos(h_angle_rad), -sin(h_angle_rad), 0;
%           sin(h_angle_rad),  cos(h_angle_rad), 0;
%           0,                 0,               1];
% 
%     % 垂直倾角 v_angle -> 绕 y 轴旋转
%     Ry = [cos(v_angle_rad), 0, sin(v_angle_rad);
%           0,                1, 0;
%           -sin(v_angle_rad), 0, cos(v_angle_rad)];
%     % y first, z second
% 
%     % 总的旋转矩阵
%     R = Rz * Ry;
% 
%     % 应用旋转矩阵到所有坐标点
%     coordinates = (R * coordinates')';
% end
% 
