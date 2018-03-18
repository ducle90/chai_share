function [ audio ] = declipCBAR( audio, isClipped, windowSize, stepSize, showProgress )
%DECLIPCBAR - Declips audio using Constrained Blind Amplitude
%Reconstruction (CBAR)
%
% Syntax:  [audio] = declipCBAR(audio,isClipped,windowSize,stepSize,showProgress)
%
% Inputs:
%    audio (Nx1 column vector) - Clipped audio signal
%    isClipped (Nx1 column vector) - Mask of clipped regions (true where clipped) 
%    windowSize (scalar) - Size of the processing window (multiple of 4) [default=80]
%    stepSize (scalar) - Size of the processing steps and inner update window 
%       (multiple of 2) [default=40]
%    showProgress (scalar) - If true, displays a processing bar notification. If 
%       closed, resutls in the algorithm terminating. [default=false]
%
% Outputs:
%    audio (Nx1 column vector) - The declipped audio 
%
% Examples (All Equivalent): 
%    declipCBAR( audio, isClipped )
%    declipCBAR( audio, isClipped, 80, 40 )
%    declipCBAR( audio, isClipped, 80, 40, false )
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Harvilla, Mark J., and Richard M. Stern. "Least squares signal declipping 
%        for robust speech recognition." Fifteenth Annual Conference of the 
%        International Speech Communication Association. 2014.
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 21-September-2015
%
% See also: declipRBAR,  declipFSE

%------------- BEGIN CODE --------------

% Check that window size exists and is a multiple of 4
if exist('windowSize','var')
    if numel(windowSize)~=1 || mod(windowSize,4)~=0
        error('windowSize must be a scalar that is a multiple of 4.');
    end
else
    windowSize = 80;
end

% Check that window size exists and is a multiple of 2
if exist('stepSize','var')
    if numel(stepSize)~=1 || mod(stepSize,2)~=0
        error('stepSize must be a scalar that is a multiple of 2.');
    end
else
    stepSize = 40;
end

% Check if showProgress declared
if ~exist('showProgress','var')
    showProgress = false;
end

% Set up waitbar
waitbarMessage = 'Declipping Using CBAR...';
if showProgress
    waitH = waitbar(0,waitbarMessage);
end

% Set up sizes
totalFrames = ceil((numel(audio)-(windowSize-stepSize))/stepSize);

% Loop though all frames for processing with update in center
for fOn = 1:totalFrames
    % Set up frame
    start = ((fOn-1)*stepSize)+1;
    stop = min(start + windowSize - 1, numel(audio));
    frame = audio(start:stop);
    frameClipped = isClipped(start:stop);

    % Set update region
    if fOn==1, selStart = 1; else selStart = (windowSize/4)+1; end
    if fOn==totalFrames, selStop = numel(frame); else selStop = ((windowSize*3)/4); end

    % Only run algorithm when update region clipped
    if any(frameClipped(selStart:selStop))
        % Set up separated signals
        Sr = eye(numel(frame));
        Sc = eye(numel(frame));
        Sr(frameClipped,:) = [];
        Sc(~frameClipped,:) = [];
        xr = Sr*frame;
        xc = Sc*frame;

        % Solve constraint optimization problem
        problem = [];
        problem.objective = @(z) sum(sqrt(abs(diff(((Sr'*xr)+(Sc'*z)),2))));
        problem.x0 = xc;
        problem.Aineq = -sign(diag(xc));
        problem.bineq = -abs(xc);
        problem.solver = 'fmincon';
        problem.options = optimoptions('fmincon','Algorithm','active-set', ...
            'Display','off');
        xc = fmincon(problem);
        y = (Sr'*xr)+(Sc'*xc);

        % Modify audio with updated declipped region
        selRange = selStart:selStop;
        audio(selRange+start-1) = y(selRange);
    end 
    
    % Update waitbar
    if showProgress
        if ishandle(waitH)
            waitbar(fOn/totalFrames,waitH,waitbarMessage);
        else
            error('Process terminated by user.');
        end
    end
end

% Close waitbar
if showProgress
    close(waitH);
end

%------------- END OF CODE --------------