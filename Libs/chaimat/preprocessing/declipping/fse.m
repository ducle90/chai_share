function [ extrapolated ] = fse( sparseSignal, isMissing, options )
%FSE - Runs Frequency Selective Extrapolation (FSE) on a sparse signal with
%missing sections in order to approximate all samples
%
% Syntax:  [extrapolated] = fse(sparseSignal,missing,options)
%
% Inputs:
%    sparseSignal (Nx1 column vector) - Signal with missing data (it doesn't
%       matter what value set to in missing regions)
%    missing (Nx1 column vector) - Mask of missing regions (true where missing) 
%    options (struct) - FSE parameters (see [1]) [optional]
%       nFFT (scalar) - Size of FFT used [default=2048]
%       rhoDecay (scalar) - The decay in weight for samples away from center [default=0.99]
%       max_iter (scalar) - The maximum amount of iterations to run without
%           convergence [default=1500]
%       odcFactor (scalar) - Orthogonality Deficiency Compensation [default=1.25]
%       dE_min (scalar) - The error threshold allowed before stopping
%           iterations [default=2]
%
% Outputs:
%    extrapolated (Nx1 column vector) - The signal estimated by FSE 
%
% Examples: 
%    fse( audio, isClipped )
%    fse( audio, isClipped, options )
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Algorithm adapted to 1D from 
%        http://www.lms.lnt.de/en/research/activity/video/vcoding/concealment.php
%    [2] Meisinger, Katrin, and André Kaup. "Spatial error concealment of 
%        corrupted image data using frequency selective extrapolation." 
%        Acoustics, Speech, and Signal Processing, 2004. Proceedings.(ICASSP'04). 
%        IEEE International Conference on. Vol. 3. IEEE, 2004.
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 21-September-2015
%
% See also: declipFSE

%------------- BEGIN CODE --------------

% Set up options if not given
if ~exist('options','var') || numel(options)==0
    options = struct();
    options.nFFT = 2048;
    options.rhoDecay = 0.99;
    options.max_iter = 1500;
    options.odcFactor = 1.25;
    options.dE_min = 2;
end

% Parameters
N = numel(sparseSignal);

% Setup vectors
sparseSignal(isMissing) = 0;
w = options.rhoDecay.^abs((1:N)'-((N-1)/2));
w(isMissing) = 0;
r = sparseSignal.*w; % Residual signal
W = fft(w,options.nFFT);
R = fft(r,options.nFFT);
G = complex(zeros(options.nFFT,1));
nFFT_complex = complex(double(options.nFFT));

% Iterate until converence to find sparse signal
for iter_counter = 1:options.max_iter
    % Determine best fitting basis function
    dE_a = abs(R);
    [~, maxInd] = max(dE_a);
    u = maxInd-1;

    % Determine expansion coefficient
    dc = options.odcFactor * R(u+1) / W(1);

    % Update the parametric model
    G(u+1) = G(u+1) + nFFT_complex * dc;

    % Check if maximal error decrease below threshold
    if dE_a(u+1) < options.dE_min
        break;
    end

    % Determine new approximation error
    R = R - dc*circshift(W,u);
end

% Return time domain
extrapolated = ifft(G);
extrapolated = real(extrapolated(1:numel(sparseSignal)));

%------------- END OF CODE --------------