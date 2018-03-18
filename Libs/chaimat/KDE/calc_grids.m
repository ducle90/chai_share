function [ X,Y ] = calc_grids( G )
%%%%
% calc_grids - calculate the grid positions of 2D kernel density estimation
% Syntax: [den_L,X,Y] = H2L_resolution(den_H, G)
%
% Inputs:
%      G (scalar) - resolution of the meshgrid output
%
% Outputs:
%    X (G*G matrix) - The x positions of grids
%    Y (G*G matrix) - The y positions of grids
%
% Toolbox required: none
% Other m-files required: none
% Subfunctions: none
%
% Author: Biqiao Zhang
% University of Michigan, Department of Computer Science and Engineering
% Email: didizbq@umich.edu
% June 2015; Last revision: 06-March-2016 (Changed function and parameter names, added comments)
%%%%%%%%%%%%%%%%%%%%%%

minxy = -1 + 1/G;
maxxy = 1 - 1/G;
scaling = maxxy - minxy;
[X,Y]=meshgrid(minxy:scaling/(G-1):maxxy,minxy:scaling/(G-1):maxxy);


end

