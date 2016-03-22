function objJaccard = ObjectJaccard(S,G)
% Jaccard index and distance co-efficient of segmemted and ground truth
% image
% Usage: [index,distance(JC)] = jaccard_coefficient(Orig_Image,Seg_Image);

S = single(S);
G = single(G);

% Check for logical image
if ~islogical(G)
    error('Image must be in logical format');
end
if ~islogical(S)
    error('Image must be in logical format');
end

listLabelS = unique(S);             % a list of labels of objects in S
listLabelS(listLabelS == 0) = [];
numS = length(listLabelS);

listLabelG = unique(G);             % a list of labels of objects in G
listLabelG(listLabelG == 0) = [];
numG = length(listLabelG);

if numS == 0 && numG == 0    % no segmented object & no ground truth objects
    objJaccard = 1;
    return 
elseif numS == 0 || numG == 0
    objJaccard = 0;
    return
else
    % do nothing
end

% calculate object-level jaccard
temp1 = 0;                          % omega_i*Jaccard(G_i,S_i)
totalAreaS = sum(S(:)>0);
for iLabelS = 1:length(listLabelS)
    Si = S == listLabelS(iLabelS);
    intersectlist = G(Si);
    intersectlist(intersectlist == 0) = [];
    
    if ~isempty(intersectlist)
        indexGi = mode(intersectlist);
        Gi = G == indexGi;
    else
        Gi = false(size(G));
    end
    
    omegai = sum(Si(:))/totalAreaS;
    temp1 = temp1 + omegai*Jaccard(Gi,Si);
end


temp2 = 0;                          % tilde_omega_i*Jaccard(tilde_G_i,tilde_S_i)
totalAreaG = sum(G(:)>0);
for iLabelG = 1:length(listLabelG)
    tildeGi = G == listLabelG(iLabelG);
    intersectlist = S(tildeGi);
    intersectlist(intersectlist == 0) = [];
    
    if ~isempty(intersectlist)
        indextildeSi = mode(intersectlist);
        tildeSi = S == indextildeSi;
    else
        tildeSi = false(size(S));
    end
    
    tildeOmegai = sum(tildeGi(:))/totalAreaG;
    temp2 = temp2 + tildeOmegai*Jaccard(tildeGi,tildeSi);
end

objJaccard = (temp1 + temp2)/2;

    function jaccard = Jaccard(A,B)
        inter_image = A&B;
        union_image = A|B;
        jaccard = sum(inter_image(:))/sum(union_image(:));
    end

end