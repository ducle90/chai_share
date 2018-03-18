% TRACLUS_segment.m

% ----------------------- %

% INPUT: 
% traj : multi-dimensional trajectory (row: time, column: feature
% dimension)
% MDL_ADVANTAGE : control sparseness of segmentation (cross validation in
% real experiments)

% OUTPUT: 
% CP: characteristic point
% realseg : each row contains [starting_point of a segment, ending_point of
% a segment] -- this can be used for the ease of future segmentation
% new_traj: final segmented trajectories in cell format

% ----------------------- %


% Created on 2014/08/11 by Yelin Kim
% Modified on 2016/3/5 by Yelin Kim

% implementation of TRACLUS segmentation of the following paper: 
% Lee, Jae-Gil, Jiawei Han, and Kyu-Young Whang. 
% "Trajectory clustering: a partition-and-group framework." 2007 ACM SIGMOD


function [CP, realseg, new_traj] =TRACLUS_segment(traj, MDL_ADVANTAGE)

pTrajectory = [[1:size(traj,1)]' medfilt1(traj,3)]; % pTrajectory: smoothed trajectory. so NaN can be transmitted. first column is time index


CheckTraj = pTrajectory(:,2:end);

new_traj = [];

% do TRACLUS partitioning

CP = 1; %add start point
realseg = [];% real segments


len = size(pTrajectory,1);

% initialization
start = 1;
seglen = 1;
cost_par = 0;
cost_nonpar = 0;

% start traclus partitioning
while (start + seglen <= len )
    
    curr = start + seglen;
    
    if curr>len
        break;
    end
    
    % if NaN value in CheckTraj(start:curr, :)
    % find until we don't have that NaN value and assign that point as the
    % new CP. INCLUDE this segment with NAN value in a segment. begin the algorithm again
    
    if sum(sum(isnan(CheckTraj(start, :)))) + sum(sum(isnan(CheckTraj(curr, :)))) > 0 % if start or current meets NaN
        while (sum(isnan(CheckTraj( curr, :)))> 0)
            curr = curr+1;
            
            if curr>len % trajectory ends with NaN value. stop here.
                new_traj = [new_traj; {traj(CP(end):end, :)}]; % add the new_traj to cell
                realseg = [realseg; [CP(end), len]];
                CP = [CP; len]; % curr-1: new CP point
                
                return;
                
            end
            
        end
        % make a NaN segment from start:curr
        % partition at the  previous point
        new_traj = [new_traj; {traj(CP(end):curr-1, :)}]; % add the new_traj to cell
        realseg = [realseg; [CP(end), curr-1]];
        CP = [CP; curr]; % curr-1: new CP point
        
        
        start = curr; seglen = 1;
        cost_nonpar = 0; % reset the cost_nonpar for the new characteristic point
        continue;
    end
    
    
    
    
    
    % display(sprintf('start: %.2f, curr:%.2f', start, curr));
    %
    % cost for partitioning and non-partitioning
    cost_par = ComputeModelCost(pTrajectory, start, curr) +... % L(H)
        ComputeEncodingCost(pTrajectory, start, curr); % L(D|H)
    
    %     display(sprintf('costpar: %.2f / L(H): %.2f, L(D|H) :%.2f', cost_par, ComputeModelCost(pTrajectory, start, curr), ComputeEncodingCost(pTrajectory, start, curr)));
    cost_nonpar = cost_nonpar + ComputeModelCost(pTrajectory, curr - 1, curr); % L(H)
    %     display(sprintf('cost_nonpar: %.2f', cost_nonpar));
    
    % check if the partitioning at the current point makes the MDL
    % cost larger than not partitioning)
    if (cost_par > cost_nonpar + MDL_ADVANTAGE)
        % display('---partition---')
        % display(sprintf('start: %.2f, curr:%.2f', start, curr));
        % display(sprintf(' cost_par(%.2f) > cost_nonpar(%.2f) + MDL_ADVANTAGE(%.2f)', cost_par, cost_nonpar, MDL_ADVANTAGE));
        % partition at the  previous point
        new_traj = [new_traj; {traj(CP(end):curr-1, :)}]; % add the new_traj to cell
        realseg = [realseg; [CP(end), curr-1]];
        CP = [CP; curr]; % curr-1: new CP point
        
        
        start = curr; seglen = 1;
        cost_nonpar = 0; % reset the cost_nonpar for the new characteristic point
    else
        seglen = seglen+1;
    end
    %     display([start+seglen len]);
    
end

% final segment
new_traj = [new_traj; {traj(CP(end):end, :)}];
realseg = [realseg; [CP(end), len]];
CP = [CP; len];


if sum( cell2mat(cellfun(@(x) size(x,1), new_traj, 'UniformOutput', false)) ) ~=len
    display('error! new_traj not cover all traj');
end
end
