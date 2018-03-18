function encodingCost = ComputeEncodingCost(traj, startInd, endInd) % L(D|H)

encodingCost = 0;

for i = startInd:endInd-1 % all possible line segment
    
    startSeg = i; % each line segment
    endSeg = i+1;
    
    %% perpendicular distance
    %     perpDist = MeasureperpDist(startInd, endInd, startSeg, endSeg);
    % perpDist
    dist1 = point_to_line(traj(startSeg,:), traj(startInd,:), traj(endInd,:));
    dist2 = point_to_line(traj(endSeg, :), traj(startInd,:), traj(endInd,:));
    
    
    % if the first line segment is exactly the same as the second one, the perpendicular distance should be zero
    if (dist1 == 0 && dist2 == 0)
        perpDist = 0;
    else
        perpDist = (dist1^2+dist2^2) / (dist1 + dist2);
        % return (d1^2 + d2^2) / (d1 + d2) as the perpendicular distance
    end
    
    %% angle distance
    %     angleDist = MeasureAngleDisntance(startInd, endInd, startSeg, endSeg);
    %     angleDist
    
    vec1 = traj(startInd,:)-traj(endInd,:);
    vec2 = traj(startSeg,:)-traj(endSeg, :);
    
    if (norm(vec1)==0 || norm(vec2) ==0)
        angleDist=0;
    else
        cosTheta = dot(vec1,vec2) / (norm(vec1)*norm(vec2));
        % compensate the computation error (e.g., 1.00001)
        % cos(theta) should be in the range [-1.0, 1.0]
        % START ...
        if (cosTheta > 1.0) cosTheta = 1.0;
        elseif (cosTheta < -1.0) cosTheta = -1.0;
        end
        
        angleDist = norm(vec2) * sqrt(1 - cosTheta^2);
    end
    
    %%
    
    
    %     if (perpDist < 1.0)
    %         perpDist = 1.0;	% to take logarithm
    %     end
    %
    %     if (angleDist < 1.0)
    %         angleDist = 1.0;	% to take logarithm
    %     end
    %
    encodingCost = encodingCost + (log2(perpDist) + log2(angleDist));
    
    
end
end
