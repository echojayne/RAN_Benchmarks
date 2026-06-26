function filter_flag = check_estimate_angles(v_mimo_match_struct, sqlite_struct, channel_id, v_match_table)
if nargin < 4
    v_match_table = [];
end
path_v_ch = sqlite_struct.path_v(sqlite_struct.path_v.channel_id == channel_id, :);
if ~isempty(v_match_table)
    path_v_ch = path_v_ch(ismember(path_v_ch.path_id, v_match_table), :);
end
aod_phi_max = get_max_diff(v_mimo_match_struct.mimo_aod_phi(:,1,1)/pi*180, path_v_ch.departure_phi);
aod_theta_max = get_max_diff(v_mimo_match_struct.mimo_aod_theta(:,1,1)/pi*180, path_v_ch.departure_theta);

aoa_phi_max = get_max_diff(v_mimo_match_struct.mimo_aoa_phi(:,1,1)/pi*180, path_v_ch.arrival_phi);
aoa_theta_max = get_max_diff(v_mimo_match_struct.mimo_aoa_theta(:,1,1)/pi*180, path_v_ch.arrival_theta);

% fprintf('Estimated Difference in AOD: phi: %f, theta: %f\n', aod_phi_max, aod_theta_max);
% fprintf('Estimated Difference in AOA: phi: %f, theta: %f\n', aoa_phi_max, aoa_theta_max);

max_diff_aod = max(aod_phi_max, aod_theta_max);
max_diff_aoa = max(aoa_phi_max, aoa_theta_max);

max_diff = max(max_diff_aod, max_diff_aoa);
filter_flag = false;
if max_diff > 30
    disp("Reflection is too Near, filtered.")
    filter_flag = true;
end
end



function max_diff = get_max_diff(A,B)

if size(A)~=size(B)
    error('A and B must have the same size');
end

max_diff = max(abs(A-B));

end