
function modify_antenna_gain(filepath)
    % Open the file for reading
    fid = fopen(filepath, 'r');
    if fid == -1
        error('File could not be opened.');
    end
    
    % Read the file into cell array
    lines = {};
    tline = fgetl(fid);
    while ischar(tline)
        lines{end+1} = tline;
        tline = fgetl(fid);
    end
    fclose(fid);
    
    % Open the file again for writing
    fid = fopen(filepath, 'w');
    if fid == -1
        error('File could not be opened.');
    end
    
    % Write the first 17 lines back without modification
    for i = 1:17
        fprintf(fid, '%s\n', lines{i});
    end
    
    % Process and modify the antenna gain information from line 18 onwards
    for i = 18:length(lines)
        % Parse the numerical data in each line
        data = sscanf(lines{i}, '%f');
        
        % If the line contains less than 6 columns, skip processing
        if length(data) < 6
            fprintf(fid, '%s\n', lines{i});
            continue;
        end
        
        % Extract the third and fourth columns
        a3 = data(3);
        a4 = data(4);
        
        % Compute the new values for the fifth and sixth columns
        new_value = (max(a3, -50) + max(a4, -50)) * 0.2;
        
        % Modify the fifth and sixth columns
        if data(5)~=0
            data(5) = data(5) + new_value*(0.9+0.2*rand());
        else
            data(5) = data(5) + 2*rand();
        end
        if data(6)~=0
            data(6) = data(6) + new_value*(0.9+0.2*rand());
        else
            data(6) = data(6) + 2*rand();
        end
        
        
        % Write the modified line back to the file
        fprintf(fid, '%.6f %.6f %.6f %.6f %.6f %.6f\n', data(1), data(2), data(3), data(4), data(5), data(6));
    end
    
    % Close the file
    fclose(fid);
end
