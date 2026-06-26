




% old version is as below
% *************************************************************************

% function MicroPatchInfo(freq_list)
% subcarriers_num = length(freq_list);
% gain_indb_box = zeros(subcarriers_num, 2*90+1, 360);
% phs_box = zeros(subcarriers_num, 2*90+1, 360);
% 
% for ii = 1:subcarriers_num
%     freq_ii = freq_list(ii);
%     antenna = design(patchMicrostrip,freq_ii);
%     phs_box(ii,:,:) = pattern(antenna, freq_ii,1:360,-90:90,'Type', 'phase', 'Polarization', 'V');
%     gain_indb_box(ii,:,:) = pattern(antenna, freq_ii,1:360,-90:90,'Type', 'directivity', 'Polarization', 'V');
%     disp(['Finished: ', num2str(ii), '/',num2str(subcarriers_num)])
% end
% 
% save('phs_box.mat','phs_box');
% save('gain_indb_box.mat', 'gain_indb_box');
% 
% end
