function progressBar(iter, total, msg)
    % progressBar 在命令行中显示进度条
    % 输入：
    %   iter  - 当前迭代次数
    %   total - 总的迭代次数
    %   msg   - 可选，进度条前的附加消息

    % 如果总迭代次数为零，则不显示进度条
    if total == 0
        return;
    end

    % 计算进度百分比
    percent = iter / total;
    
    % 显示进度条
    progress = round(percent * 50);  % 进度条长度为 50 个字符
    bar = repmat('=', 1, progress);  % 完成部分
    spaces = repmat(' ', 1, 50 - progress);  % 未完成部分
    fprintf([msg, ': [', bar, spaces, '] %.2f%%\r'], percent * 100);
    
    % 刷新输出，`\r` 回退到行首
    if iter == total
        fprintf('\n');  % 在进度条结束时换行
    end
end
