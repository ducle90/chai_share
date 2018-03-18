function [ audio ] = declipRBAR( audio, isClipped, lambda, windowSize, stepSize, showProgress )
%DECLIPRBAR - Declips audio using Regularized Blind Amplitude
%Reconstruction (RBAR)
%
% Syntax:  [audio] = declipRBAR(audio,isClipped,lambda,windowSize,stepSize,showProgress)
%
% Inputs:
%    audio (Nx1 column vector) - Clipped audio signal
%    isClipped (Nx1 column vector) - Mask of clipped regions (true where clipped) 
%    lambda (scalar) - Regularization parameter (see [1]) [default=0.1]
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
%    declipRBAR( audio, isClipped )
%    declipRBAR( audio, isClipped, 0.1 )
%    declipRBAR( audio, isClipped, 0.1, 80, 40 )
%    declipRBAR( audio, isClipped, 0.1, 80, 40, false )
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% References:
%    [1] Harvilla, Mark J., and Richard M. Stern. "EFFICIENT AUDIO DECLIPPING 
%        USING REGULARIZED LEAST SQUARES."
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 21-September-2015
%
% See also: declipCBAR,  declipFSE

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
waitbarMessage = 'Declipping Using RBAR...';
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
        
        % Set up variables for RBAR approximation
        isPos = sign(xc)==1;
        Scp = eye(numel(xc));
        Scn = eye(numel(xc));
        Scp(~isPos,:) = [];
        Scn(isPos,:) = [];
        xcp = Scp*xc;
        xcn = Scn*xc;
        
        % Determine parameters using equations from paper [1]
        pClipped = numel(xc)/numel(frame);
        if pClipped <= 0.9, phi = exp(2.481*pClipped);
        else phi = (271.7493*(pClipped^59.9519))+8.8361;
        end
        t0 = xcp*phi;
        t1 = -xcn*phi;

        % Solve closed form RBAR solution
        xc = (-inv((diff(Sc,2,2)*diff(Sc,2,2)')+(lambda*((Scp'*Scp)+(Scn'*Scn)))))* ...
            ((diff(Sc,2,2)*diff(Sr,2,2)'*xr)-(lambda*((Scp'*t0)-(Scn'*t1))));
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