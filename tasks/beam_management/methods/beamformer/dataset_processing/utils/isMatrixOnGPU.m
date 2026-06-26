function isOnGPU = isMatrixOnGPU(matrix)
    % isMatrixOnGPU 判断输入的矩阵是否在 GPU 上
    %
    % 输入:
    %   matrix - 一个矩阵，可能是 GPU 数组或 CPU 数组
    %
    % 输出:
    %   isOnGPU - 如果矩阵在 GPU 上，则返回 true，否则返回 false
    
    % 使用 isa 函数检查输入是否为 GPU 数组
    isOnGPU = isa(matrix, 'gpuArray');
    
    % 打印详细信息
    if isOnGPU == false
        disp('The matrix is NOT in CPU');
    end
end
