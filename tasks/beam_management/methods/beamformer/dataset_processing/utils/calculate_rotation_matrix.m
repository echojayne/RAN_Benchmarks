
function R = calculate_rotation_matrix(theta_y, phi_z)
% Input should be in rad
    % y 轴旋转矩阵
    R_y = [cos(theta_y), 0, sin(theta_y);
        0, 1, 0;
        -sin(theta_y), 0, cos(theta_y)];
    % z 轴旋转矩阵
    R_z = [cos(phi_z), -sin(phi_z), 0;
        sin(phi_z), cos(phi_z), 0;
        0, 0, 1];
    % 先沿 y 轴旋转，再沿 z 轴旋转
    R = R_z * R_y;
end



% function R = calculate_rotation_matrix(old, new)
% % v:3*1
% % 归一化输入向量 v
% old = old / norm(old);
% new = new / norm(new);
% 
% 
% if isequal(old, new)
%     R = eye(3);
%     return;
% end
% 
% % 计算旋转轴 (叉积)
% 
% rotation_axis = cross(old, new);
% rotation_axis = rotation_axis / norm(rotation_axis); % 归一化
% 
% % 计算旋转角度 (点积)
% theta = acos(dot(old, new));
% 
% % 构造旋转矩阵（绕 rotation_axis 旋转 theta 角度）- Rodrigues' rotation formula
% K = [ 0, -rotation_axis(3), rotation_axis(2);
%     rotation_axis(3), 0, -rotation_axis(1);
%     -rotation_axis(2), rotation_axis(1), 0];
% R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K);
% % 
% % y_axis = R * [0; 1; 0];
% 
% end
