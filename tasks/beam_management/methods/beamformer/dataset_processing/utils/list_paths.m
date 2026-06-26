function path_list = list_paths(parent_folder_path, pod_id, total_server_num)
    path_list = {};
    folder_content = dir(parent_folder_path);
    
    % 过滤出所有目录
    folders = folder_content([folder_content.isdir] & ~strcmp({folder_content.name}, '.') & ~strcmp({folder_content.name}, '..'));
    
    % 计算每个目录的下标
    for i = 1:length(folders)
        if mod(i - 1, total_server_num) + 1 == pod_id
            folder_path = fullfile(parent_folder_path, folders(i).name);
            path_list{end+1} = folder_path;
        end
    end
end
