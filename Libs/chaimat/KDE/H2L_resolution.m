function [den_L,X,Y] = H2L_resolution(den_H, G)
%%%%
% H2L_resolution - High to low resulution of 2D kernel density estimation, 
%                  normalized to be a probability distribution
% Syntax: [den_L,X,Y] = H2L_resolution(den_H, G)
%
% Inputs:
%      den_H (Gh*Gh matrix, sum to 1 in total) - High resolution density probability distribution
%      G (scalar, must be exact division of Gh ) - resolution of the
%      meshgrid output
%
% Outputs:
%    den_L (G*G matrix) - Low resolution density probability distribution
%    X (G*G matrix) - The x positions of grids in low resolution density
%    Y (G*G matrix) - The y positions of grids in low resolution density
%
% Toolbox required: none
% Other m-files required: calc_grids.m
% Subfunctions: calc_grids.m
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% June 2015; Last revision: 06-March-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

G_orig = length(den_H);
if mod(G_orig,G) ~= 0
    error('G must be exact division of the original resolution (size of den_H)')
end

d = G_orig/G;
den_L = zeros(G,G);
for i=1:G
    for j=1:G
        den_L(i,j)=sum(sum(den_H(i*d-d+1:i*d,j*d-d+1:j*d)));
    end
end

den_L = den_L./sum(sum(den_L));

[X,Y] = calc_grids(G);

end

