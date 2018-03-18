function [ bandwidth,density,X,Y ] = kde2d_prob_distibution( labels, G )
%%%%
% kde2d_prob_distibution - 2-D kernel density estimation, normalized to be a
% probability distribution
% Syntax: [ bandwidth,density,X,Y ] = kde2d_prob_distibution( labels, G )
%
% Inputs:
%      labels (N*2 matrix, double) - 2-D continious labels of the data
%      G (scalar, must be power of 2) - resolution of the meshgrid (G*G)
%
% Outputs:
%    bandwidth (1*2 matrix) - optimal bandwidths for a bivaroate Gaussian kernel
%    density (G*G matrix) - probability of each grid area
%    X (G*G matrix) - The x positions of grids in density
%    Y (G*G matrix) - The y positions of grids in density
%
% Toolbox required: none
% Other m-files required: kde2d.m (http://www.mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation)
% Subfunctions: kde2d.m

% References:
%    [1] Kernel density estimation via diffusion
%        Z. I. Botev, J. F. Grotowski, and D. P. Kroese (2010)
%        Annals of Statistics, Volume 38, Number 5, pages 2916-2957.
%        Conference on Machine Learning (ICML-11). 2011.
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% June 2015; Last revision: 06-March-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

if mod(log2(G),1) ~= 0
    error('G must be power of 2.')
end

if size(labels,2) ~= 2
    error('Labels must be N*2.')
end

xymin = -1 + 1/G;
xymax = 1 - 1/G;
[bandwidth,density,X,Y] = kde2d(labels,G,[xymin xymin],[xymax xymax]);
density = density./sum(sum(density));

end

