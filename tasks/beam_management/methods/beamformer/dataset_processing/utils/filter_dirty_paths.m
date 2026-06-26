function sqlite_struct = filter_dirty_paths(sqlite_struct)

dirty_paths = sqlite_struct.path_v.path_id(sqlite_struct.path_v.departure_theta > 150);

sqlite_struct.path_v = sqlite_struct.path_v(sqlite_struct.path_v.departure_theta <= 150, :);

fprintf('Filter: %d paths\n', length(dirty_paths));

rows_to_keep = ~ismember(sqlite_struct.path_utd_v(:,1), dirty_paths);
sqlite_struct.path_utd_v = sqlite_struct.path_utd_v(rows_to_keep, :);

rows_to_keep = ~ismember(sqlite_struct.interaction.path_id, dirty_paths);
sqlite_struct.interaction = sqlite_struct.interaction(rows_to_keep,:);
end
