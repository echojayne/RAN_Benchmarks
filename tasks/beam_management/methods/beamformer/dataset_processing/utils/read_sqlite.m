function res_struct = read_sqlite(sqlite_folder,  splite_num)
res_struct = struct();
res_struct.is_valid = true; 
sqlite_H_path = auto_find_sqlite(sqlite_folder, 'tx', 'h');
sqlite_V_path = auto_find_sqlite(sqlite_folder, 'tx', 'v');
try
    conn_V = sqlite(sqlite_V_path, 'readonly');
    conn_H = sqlite(sqlite_H_path, 'readonly');
    quiry_intercation = "SELECT path_id,interaction_type_id,x,y,z FROM interaction";
    quiry_path = "SELECT * FROM path";
    quiry_rx = "SELECT * FROM rx";
    quiry_tx = "SELECT * FROM tx";

    res_struct.interaction = fetch(conn_V, quiry_intercation);
    res_struct.tx = fetch(conn_V, quiry_tx);
    res_struct.rx = fetch(conn_V, quiry_rx);
    res_struct.path_v = fetch(conn_V, quiry_path);
    res_struct.path_h = fetch(conn_H, quiry_path);
    res_struct.path_utd_v = read_large_sqlite(sqlite_V_path,  splite_num);
    res_struct.path_utd_h = read_large_sqlite(sqlite_H_path,  splite_num);
    conn_V.close();
    conn_H.close();
catch ME
    warning("Failed to read sqlite files in %s: %s", sqlite_folder, ME.message);
    res_struct.is_valid = false;
    % 避免 conn 未定义报错
    try, conn_V.close(); end
    try, conn_H.close(); end
end
end

function sqlite_data = read_large_sqlite(sqlite_path,  splite_num)

conn = sqlite(sqlite_path, 'readonly');
len_quiry = 'SELECT COUNT(*) FROM path_utd';
total_len = table2array(fetch(conn,len_quiry));

sqlite_data = zeros(total_len, 6);
segments = divide_sequence(total_len,splite_num);

for i = 1:splite_num
    query = sprintf('SELECT * FROM path_utd WHERE path_utd_id BETWEEN %d AND %d', segments{i}(1), segments{i}(2));
    data = fetch(conn,query);
    sqlite_data(segments{i}(1):segments{i}(2),:) = [cast(data.path_id,'double'),...
                                                    cast(data.utd_instance_id, 'double'),...
                                                    data.e_theta_r, data.e_theta_i, data.e_phi_r, data.e_phi_i];
end

conn.close();

end

function segments = divide_sequence(N,splite_num)
    N = cast(N, 'double');
    indices = round(linspace(1, N+1, splite_num+1));  % 均匀生成边界索引
    
    segments = cell(1, splite_num);  % 初始化分段
    for i = 1:splite_num
        segments{i} = [indices(i),(indices(i+1)-1)];  % 每一段的索引范围
    end
end

function file_path = auto_find_sqlite(folder, prefix, suffix)
    candidates = { ...
        fullfile(folder, sprintf('%s_%s.sqlite', prefix, suffix)), ...
        fullfile(folder, sprintf('%s-%s.sqlite', prefix, suffix)) ...
    };
    
    for i = 1:numel(candidates)
        if exist(candidates{i}, 'file')
            file_path = candidates{i};
            return;
        end
    end
    
    error('找不到对应的sqlite文件：%s_{h,v}.sqlite 或 %s-{h,v}.sqlite', prefix, prefix);
end
