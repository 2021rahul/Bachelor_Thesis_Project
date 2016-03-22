TP=zeros(13,1);
FP=zeros(13,1);
FN=zeros(13,1);
for i=0:12
    im = strcat(int2str(i), '.jpg');
    anno = strcat(int2str(i),'_anno.jpg');
    im = strcat('E:\academics\SEMESTER_7\BTP\data_normalized\result_maps\', im);
    anno = strcat('E:\academics\SEMESTER_7\BTP\data_normalized\result_maps\', anno);
    
    S = imread(im);
    G = imread(anno);
    S(S>127)=255;
    S(S<127)=0;
    G(G>127)=255;
    G(G<127)=0;
    S = S/255;
    G = G/255;
    
    [tp, fp, fn] = F1score(S,G);
    TP(i+1,1) = tp;
    FP(i+1,1) = fp;
    FN(i+1,1) = fn;
    
    
end
 precision = TP./(TP + FP);
 recall = TP./(TP + FN);
 score = (2*precision.*recall)./(precision+recall);