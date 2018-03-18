%TESTDECLIPPING - Tests all included declipping methods and displays
%comparison plots
%
% Other m-files required: declipCBAR, declipRBAR, declipFSE, fse
% Subfunctions: none
% MAT-files required: none
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 22-September-2015
%
% See also: declipCBAR,  declipRBAR,  declipFSE

%------------- BEGIN CODE --------------

%% Load Audio
[audio, Fs] = audioread('/home/gideonjn/Samples/TestAudio.wav');
audio = audio(1:32000); % only use subset

%% Artificially clipped at 0.4
clipThreshold = 0.4;
isClipped = abs(audio)>=clipThreshold;
clipped = audio;
clipped(clipped>=clipThreshold) = clipThreshold;
clipped(clipped<=-clipThreshold) = -clipThreshold;

%% Run CBAR
tic;
declippedCBAR = declipCBAR(clipped, isClipped, 80, 40, true);
timeCBAR = toc;

%% Run RBAR
tic;
declippedRBAR = declipRBAR(clipped, isClipped, 0.1, 80, 40, true);
timeRBAR = toc;

%% Run FSE
tic;
fseOptions = struct();
fseOptions.nFFT = 2048;
fseOptions.rhoDecay = 0.99;
fseOptions.max_iter = 1500;
fseOptions.odcFactor = 1.25;
fseOptions.dE_min = 2;
declippedFSE  = declipFSE( clipped, isClipped, 500, 50, fseOptions, true);
timeFSE = toc;

%% Display plots
figure;
plot([audio clipped-2 declippedCBAR-4 declippedRBAR-6 declippedFSE-8]);
legend('Original','Clipped','CBAR','RBAR','FSE');
title('Different methods of declipping');

%% Print times
figure;
bar([timeCBAR timeRBAR timeFSE]);
set(gca,'XTickLabel',{'CBAR' 'RBAR' 'FSE'})
title('Time to declip 4 seconds of audio');
xlabel('Declipping Method');
ylabel('Time to Declip (Seconds)');

%------------- END OF CODE --------------
