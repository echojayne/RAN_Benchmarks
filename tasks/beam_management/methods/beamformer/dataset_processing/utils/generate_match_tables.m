function [v_match_table, h_match_table] = generate_match_tables(sqlite_struct, channel_id)

path_by_channel_v = sqlite_struct.path_v(sqlite_struct.path_v.channel_id == channel_id,:);
path_by_channel_h = sqlite_struct.path_h(sqlite_struct.path_h.channel_id == channel_id,:);

v_match_table = path_by_channel_v.path_id;
h_match_table = zeros(size(v_match_table));

for iter = 1:length(v_match_table)
    indicator_iter = path_by_channel_v(path_by_channel_v.path_id == v_match_table(iter), "indicator");
    indicator_iter = table2array(indicator_iter);
    if isa(path_by_channel_h.indicator, class(indicator_iter)) && ...
            any(path_by_channel_h.indicator == indicator_iter)
        h_match_table(iter) = table2array( ...
            path_by_channel_h(path_by_channel_h.indicator == indicator_iter, "path_id") ...
        );
    else
        h_match_table(iter) = 0;
    end
end
end