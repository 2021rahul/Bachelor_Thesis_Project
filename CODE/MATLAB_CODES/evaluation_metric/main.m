objDice=zeros(72,1);
objJaccard=zeros(72,1);
objscore=zeros(72,1);
dice=zeros(72,1);
jaccard=zeros(72,1);
score=zeros(72,1);


for i=0:71
    im = strcat(int2str(i), '_post.bmp');
    anno = strcat(int2str(i),'_bin.bmp');
    %im = strcat('C:\Users\SURENDRA JAIN\Documents\AJ\Acads\VII sem\BTP\new_result_maps\', im);
    %anno = strcat('C:\Users\SURENDRA JAIN\Documents\AJ\Acads\VII sem\BTP\new_result_maps\', anno);
    
    S = imread(im);
    G = imread(anno);
    %         S(S>127)=255;
    %         S(S<127)=0;
    %         G(G>127)=255;
    %         G(G<127)=0;
    %         S = S/255;
    %         G = G/255;
    bw = bwlabel(S,8);
    bwa = bwlabel(G,8);
    
    objscore(i+1,1) = F1score(bw,bwa);
    objDice(i+1,1) = ObjectDice(bw,bwa);
    objJaccard(i+1,1) = ObjectJaccard(bw,bwa);
    score(i+1,1) = F1score_pixel(S,G);
    dice(i+1,1) = Dice(S,G);
    jaccard(i+1,1) = Jaccard(S,G);
    
end

%  precision = TP./(TP + FP);
%  recall = TP./(TP + FN);
%
%  score = (2*precision.*recall)./(precision+recall);