objDice=zeros(13,1);
objHausdorff=zeros(13,1);
score=zeros(13,1);

for i=0:12
    im = strcat(int2str(i), '_post.jpg');
    anno = strcat(int2str(i),'_anno.jpg');
    im = strcat('C:\Users\SURENDRA JAIN\Documents\AJ\Acads\VII sem\BTP\new_result_maps\', im);
    anno = strcat('C:\Users\SURENDRA JAIN\Documents\AJ\Acads\VII sem\BTP\new_result_maps\', anno);
    
    S = imread(im);
    G = imread(anno);
    S(S>127)=255;
    S(S<127)=0;
    G(G>127)=255;
    G(G<127)=0;
    S = S/255;
    G = G/255;
    bw = bwlabel(S,8);
    bwa = bwlabel(G,8);
     score(i+1,1) = F1score(bw,bwa);
%     TP(i+1,1) = tp;
%     FP(i+1,1) = fp;
%     FN(i+1,1) = fn;
    objDice(i+1,1) = ObjectDice(bw,bwa);
    objHausdorff(i+1,1) = ObjectHausdorff(bw,bwa);
    
end

%  precision = TP./(TP + FP);
%  recall = TP./(TP + FN);
% 
%  score = (2*precision.*recall)./(precision+recall);