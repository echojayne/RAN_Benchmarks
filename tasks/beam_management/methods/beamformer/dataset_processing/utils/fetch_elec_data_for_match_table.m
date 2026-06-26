function elec_match_data = fetch_elec_data_for_match_table(path_utd, match_table, subcarriers_num)

% path_utd:
% path_id, utd_instance_id, e_theta_r, e_theta_i, e_phi_r, e_phi_i
[~, idx] = ismember(path_utd(:,1),match_table);
select_data = path_utd(idx>0, :);

valid_path_num = length(match_table);
elec_match_data = zeros(valid_path_num,subcarriers_num,6);
for item = 1:length(match_table)
    path_id = match_table(item);
    if path_id == 0
        continue;
    end
    useful_data = select_data(select_data(:,1)==path_id, :);% useful_data may be not 384-rows
    useful_index = useful_data(:,2) + 1;
    elec_match_data(item,useful_index,:) =  useful_data;
end
end