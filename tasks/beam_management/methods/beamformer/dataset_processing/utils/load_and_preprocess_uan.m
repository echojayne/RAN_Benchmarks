function preloaded_data = load_and_preprocess_uan(uan_path)
    % 读取UAN文件，假设文件每行6列数据：[theta, phi, gain(theta), gain(phi), phase(theta), phase(phi)]
    opts = detectImportOptions(uan_path, 'FileType','text', 'NumHeaderLines', 17);  % 如果文件没有标题行
    opts.Delimiter = ' ';  % 假设使用空格作为分隔符，可以根据需要更改

    % 读取文件并转换为矩阵形式
    data_table = readtable(uan_path, opts);
    data = table2array(data_table);  % 将表格转换为矩阵
    
    % 提取数据列：theta, phi, gain (theta), gain (phi), phase (theta), phase (phi)
    theta = data(:, 1);
    phi = data(:, 2);
    gain_theta = data(:, 3);
    phase_theta = data(:, 5);
    
    % 创建唯一的theta和phi
    theta_unique = unique(theta);
    phi_unique = unique(phi);
    
    % 将增益和相位数据转换成网格形式
    gain_theta_grid = reshape(gain_theta, length(phi_unique), length(theta_unique));
    phase_theta_grid = reshape(phase_theta, length(phi_unique), length(theta_unique));

    % 保存预处理后的数据
    preloaded_data.theta_unique = theta_unique;
    preloaded_data.phi_unique = phi_unique;
    preloaded_data.gain_theta_grid = gain_theta_grid;
    preloaded_data.phase_theta_grid = phase_theta_grid;
end
