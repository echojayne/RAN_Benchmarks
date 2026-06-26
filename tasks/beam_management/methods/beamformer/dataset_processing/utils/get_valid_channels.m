function [channel_num, channel_list, angle_per_channel] = get_valid_channels(path_v)
    channel_list = unique(path_v.channel_id);
    channel_num = length(channel_list);
    angle_per_channel = extract_angles(path_v);
end


function angle_per_channel = extract_angles(path_v)
    % 提取每个 channel 的第一个 path 的 indicator 数值
    % 假设 path_v 按 path_id 已经排序
    
    % 获取所有 channel_id 序列
    all_channel_ids = path_v.channel_id;

    % 找出每个 channel_id 第一次出现的位置
    [unique_channel_ids, first_indices] = unique(all_channel_ids, 'first');

    % 初始化结果结构体
    max_channel_id = max(unique_channel_ids);
    angle_per_channel = struct('aod_phi', cell(1, max_channel_id), ...
                               'aod_theta', cell(1, max_channel_id), ...
                               'aoa_phi', cell(1, max_channel_id), ...
                               'aoa_theta', cell(1, max_channel_id));

    % 填充结果
    for i = 1:length(unique_channel_ids)
        ch_id = unique_channel_ids(i);
        idx = first_indices(i);

        angle_per_channel(ch_id).aod_phi   = path_v.departure_phi(idx);
        angle_per_channel(ch_id).aod_theta = path_v.departure_theta(idx);
        angle_per_channel(ch_id).aoa_phi   = path_v.arrival_phi(idx);
        angle_per_channel(ch_id).aoa_theta = path_v.arrival_theta(idx);
    end
end
