%TESTCOMBOSAD - Tests ComboSAD segmentation and displays plots
%
% Other m-files required: extractComboSAD, resampleSignalAfterWindowing,
%                         resampleTimesAfterWindowing, formSegments,
%                         enframe (from VOICEBOX toolbox)
% Subfunctions: none
% MAT-files required: none
%
% Author: John Gideon
% University of Michigan, Department of Computer Science and Engineering
% Email: gideonjn@umich.edu
% September 2015; Last revision: 29-September-2015
%
% See also: extractComboSAD,  resampleSignalAfterWindowing, 
%           resampleTimesAfterWindowing, formSegments

%------------- BEGIN CODE --------------

%% Load Audio
[audio, Fs] = audioread('/home/gideonjn/Samples/FightClub.wav');

%% Play Audio
player = audioplayer(audio,Fs);
play(player);

%% Extract ComboSAD Signal
[comboSignal, segParams] = extractComboSAD(audio, Fs);

%% Resample windowed signal and normalize
vad = resampleSignalAfterWindowing(comboSignal, segParams);
vad = (vad-prctile(vad,5))./std(vad);

%% Threshold segmentation signal and remove non-speech
onlySpeech = audio(vad>1.3);

%% Play only speech (not great for rhythm features)
player = audioplayer(onlySpeech,Fs);
play(player);

%% Form into larger more contiguous segments
Fss = segParams.Fss; % Sample rate of segmentation signal
Segments = formSegments(comboSignal, 1.6, round(0.25*Fss), round(0.7*Fss));

%% Resample windowed times
Segments = resampleTimesAfterWindowing(Segments, segParams);

%% Stitch together segments
speechSegments = stitchSignalBySegments(audio, Segments);

%% Play stitched speech segments
player = audioplayer(speechSegments,Fs);
play(player);

%% Display different signals
contSig = zeros(size(audio));
pos = cell2mat(arrayfun(@(z) Segments.Start(z):Segments.Stop(z), ...
    1:height(Segments), 'UniformOutput', false));
contSig(pos) = 1;
plot([audio vad contSig]);
xlim([1 numel(audio)]);
legend('Audio','Segmentation Signal','Contiguous Segments','Location','southeast')

%------------- END OF CODE --------------
