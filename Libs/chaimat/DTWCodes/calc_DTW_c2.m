% Created by Yelin Kim - on 2015/10/4
% Modified by Yelin Kim - on 2016/3/5

function Dist_matrix = calc_DTW_c2(input, refer) %#codegen

Dist_matrix = zeros(length(input), length(refer), 6);

if isempty(input) || isempty(refer)
    Dist_matrix = [];
else
    
    for j = 1:length(refer)
        % removed for loop (for i = 1:length(input)) to speed up.
        FacDistAll = cellfun(@(x, y) TwoTrajDTW(x, y),...
            repmat(refer(j,1), length(input), 1), input, 'UniformOutput', false);
        
        Dist_matrix(:, j, : ) = cell2mat(FacDistAll);
        
    end
end
end

