function FaceDist = TwoTrajDTW(TT, LL)    
y = [126/2+1:126];
CHI_idx = [12, 13, 14, y([12, 13, 14])];
MOU_idx = [52, 54, 58, 33, 34, 55, 57, 43, y([52, 54, 58, 33, 34, 55, 57, 43 ])];
CHK_idx = [44, 45, 46, 18, 49, 47, 48, 50 , 35, 36, 37, 38, 40, 8, 39, 41,...
    y([ 44, 45, 46, 18, 49, 47, 48, 50 , 35, 36, 37, 38, 40, 8, 39, 41 ])];
BRO_idx = [19, 24, 25, 59,60, 28, 29, 7,...
    y([ 19, 24, 25, 59,60, 28, 29, 7 ])];
BM_idx = [1, 2, 20, 21, 4, 5, 6, 23, ...
    y([ 1, 2, 20, 21, 4, 5, 6, 23 ])];
FH_idx = [61, 62, 63, y([61,62,63])];


%%
        test_CHI = TT(:, CHI_idx);
        test_FH = TT(:,FH_idx);
        test_CHK = TT(:,CHK_idx);
        test_BM = TT(:,BM_idx);
        test_BRO = TT(:,BRO_idx);
        test_MOU = TT(:,MOU_idx);
        
        ref_CHI = LL(:,CHI_idx);
        ref_FH = LL(:,FH_idx);
        ref_CHK = LL(:,CHK_idx);
        ref_BM = LL(:,BM_idx);
        ref_BRO = LL(:,BRO_idx);
        ref_MOU = LL(:,MOU_idx);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        dist_CHI = dtw_c(test_CHI', ref_CHI'); % chin: 9
        dist_FH= dtw_c(test_FH', ref_FH'); %  forehead: 9
        dist_CHK = dtw_c(test_CHK', ref_CHK'); % cheek: 48
        %  [dist_LLID_RLID, D, ~, w] = dtw_new(TT(67:72), LL(67:72)); %eyelid
        % [dist_NOSE, D, ~, w] = dtw_new(TT(73:87), refer{j,1}(73:87)); % nose
        dist_BM = dtw_c(test_BM', ref_BM'); % upper brow: 24
        dist_BRO = dtw_c(test_BRO', ref_BRO'); % eyebrows: 24
        dist_MOU = dtw_c(test_MOU', ref_MOU'); % mouth: 24
        %   [dist_HEAD, D, ~, w] = dtw_new(TT(160:165), refer{j,1}(160:165));    % head
        
        %        Dist_matrix{i, j} = [dist_CHI, dist_FH, dist_CHK, dist_BM, dist_BRO, dist_MOU];
        
        % divide by frame numbers
        n = size(test_CHI,1);
        m = size(ref_CHI,1);
        %             fram = n*m;
        
        FaceDist = zeros(1,1,6);
        FaceDist(1,1, :) = [dist_CHI, ...
            dist_FH, ...
            dist_CHK, ...
            dist_BM, ...
            dist_BRO, ...
            dist_MOU]./(n+m);

end
