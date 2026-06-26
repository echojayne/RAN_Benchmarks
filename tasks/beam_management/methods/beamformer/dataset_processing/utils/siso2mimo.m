function polar_component_complex_mimo = siso2mimo(polar_component_complex_siso, mimo_complex)
% mimo_complex: num_path, num_freq,num_rx,num_tx
num_path = size(mimo_complex,1);
num_freq = size(mimo_complex,2);
num_rx = size(mimo_complex,3);
num_tx = size(mimo_complex,4);

polar_component_complex_siso_updim = reshape(polar_component_complex_siso, ...
    [size(polar_component_complex_siso), 1, 1]);
polar_component_complex_siso_broadcast = repmat(polar_component_complex_siso_updim, ...
    [1,1,num_rx, num_tx]);
% polar_component_complex_mimo = complex(zeros(size(mimo_complex)));
% for path_idx = 1:num_path
%     polar_component_complex_mimo(path_idx,:,:,:) = polar_component_complex_siso_broadcast(path_idx,:,:,:).*mimo_complex(path_idx,:,:,:);
% end
polar_component_complex_mimo = polar_component_complex_siso_broadcast.*mimo_complex;
end

