function model_cost = ComputeModelCost(traj, startInd, endInd) % L(H)

distance = sqrt(sum((traj(endInd,:) - traj(startInd,:)).^2)); % same as norm(traj(endInd,:) - traj(startInd,:)), but a bit faster


% if (distance < 1.0)
%     distance = 1.0;		% to take logarithm
% end

model_cost =  log2(distance);

if isnan(model_cost)
    display('error in ComputModelCost!');
end
end
