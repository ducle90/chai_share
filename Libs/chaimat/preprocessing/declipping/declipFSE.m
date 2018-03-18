function [ audio ] = declipFSE( audio, isClipped, windowSize, stepSize, fseOptions, showProgress)
%DECLIPFSE - Declips audio using Frequency Selective Extrapolation (FSE)
%
% Syntax:  [audio] = declipFSE(audio,isClipped,windowSize,stepSize,fseOptions,showProgress)
%
% Inputs:
%    audio (Nx1 column vector) - Clipped audio signal
%    isClipped (Nx1 column vector) - Mask of clipped regions (true where clipped) 
%    windowSize (scalar) - Size of the processing window (multiple of 4) [default=500]
%    stepSize (scalar) - Size of the processing steps and inner update window 
%       (multiple of 2) [default=50]
%    fseOptions (struct) - Parameters for FSE algorithm (see fse and [1]) [optional]
%    showProgress (scalar) - If true, displays a processing bar notification. If 
%       closed, resutls in the algorithm terminating. [default=false]
%
% Outputs:
%    audio (Nx1 column vector) - The declipped audio 
%
% Examples (All Equivalent): 
%    declipFSE(audio,isClipped)
%    declipFSE(audio,isClipped,500,50)
%    declipFSE(audio,isClipped,500,50,options,false)
%
% Other m-files required: fse
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Jonscher, Markus, Juergen Seiler, and Andre Kaup. "Declipping of 
%        Speech Signals Using Frequency Selective Extrapolation." Speech 
%        Communication; 11. ITG Symposium; Proceedings of. VDE, 2014.
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 21-September-2015
%
% See also: declipRBAR,  declipCBAR,  fse

%------------- BEGIN CODE --------------

% Check that window size exists and is a multiple of 4
if exist('windowSize','var')
    if numel(windowSize)~=1 || mod(windowSize,4)~=0
        error('windowSize must be a scalar that is a multiple of 4.');
    end
else
    windowSize = 500;
end

% Check that window size exists and is a multiple of 2
if exist('stepSize','var')
    if numel(stepSize)~=1 || mod(stepSize,2)~=0
        error('stepSize must be a scalar that is a multiple of 2.');
    end
else
    stepSize = 50;
end

% Check if fseOptions declared
if ~exist('fseOptions','var')
    fseOptions = [];
end

% Check if showProgress declared
if ~exist('showProgress','var')
    showProgress = false;
end

% Set up waitbar
waitbarMessage = 'Declipping Using FSE...';
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
        y = fse(frame, frameClipped, fseOptions);
        frame(frameClipped) = y(frameClipped);

        % Modify audio with updated declipped region
        selRange = selStart:selStop;
        audio(selRange+start-1) = frame(selRange);
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