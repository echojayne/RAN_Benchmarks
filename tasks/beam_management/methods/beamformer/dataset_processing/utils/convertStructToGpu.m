function output = convertStructToGpu(input)
    % 如果输入不是结构体，直接返回
    if ~isstruct(input)
        output = input;
        return;
    end
    
    % 创建一个新的结构体，用于存储 GPU 数据
    output = struct();
    
    % 遍历输入结构体中的每一个字段
    fieldNames = fieldnames(input);
    for i = 1:numel(fieldNames)
        fieldName = fieldNames{i};
        fieldValue = input.(fieldName);
        
        % 如果该字段是数组，将其转换为 gpuArray
        if isnumeric(fieldValue)
            output.(fieldName) = gpuArray(fieldValue);
        elseif isstruct(fieldValue)
            % 如果该字段是结构体，递归处理
            error("Struct Error, code 19929309");
        else
            % 如果该字段不是数组或结构体，保持原样
            output.(fieldName) = fieldValue;
        end
    end
end
