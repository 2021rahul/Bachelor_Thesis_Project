function [TP,FP,FN] = F1score_pixel(S,G)
S = single(S);
G = single(G);
temp = S & G;
TP = sum(sum(temp(:,:)==1));
temp = S & (~G);
FP = sum(sum(temp(:,:)==1));
temp = (~S) & G;
FN = sum(sum(temp(:,:)==1));
end
