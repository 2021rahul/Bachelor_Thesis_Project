function score = F1score_pixel(S,G)
S = single(S);
G = single(G);
temp = S & G;
TP = sum(sum(temp(:,:)==1));
temp = S & (~G);
FP = sum(sum(temp(:,:)==1));
temp = (~S) & G;
FN = sum(sum(temp(:,:)==1));

precision = TP/(TP + FP);
recall = TP/(TP + FN);

score = (2*precision*recall)/(precision+recall);
end
