objDice=zeros(78,1);
objJaccard=zeros(78,1);
objscore=zeros(78,1);
dice=zeros(78,1);
jaccard=zeros(78,1);
score=zeros(78,1);


for i=0:79
    im = strcat(int2str(i), '_post.bmp');
    anno = strcat(int2str(i),'_bin.bmp');
    im = strcat('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_a/', im);
    anno = strcat('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/Bachelor_Thesis_Project/Exp1_a/', anno);
    
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
    if i<12
        objscore(i+1,1) = F1score(bw,bwa);        
        objDice(i+1,1) = ObjectDice(bw,bwa);
        objJaccard(i+1,1) = ObjectJaccard(bw,bwa);
        score(i+1,1) = F1score_pixel(bw,bwa);        
        dice(i+1,1) = Dice(bw,bwa);
        jaccard(i+1,1) = Jaccard(bw,bwa);
    end
    if i>12 && i<76
        objscore(i,1) = F1score(bw,bwa);        
        objDice(i,1) = ObjectDice(bw,bwa);
        objJaccard(i,1) = ObjectHauJaccard(bw,bwa);
        score(i,1) = F1score_pixel(bw,bwa);        
        dice(i,1) = Dice(bw,bwa);
        jaccard(i,1) = Jaccard(bw,bwa);
    end        
    if i>76
        objscore(i-1,1) = F1score(bw,bwa);        
        objDice(i-1,1) = ObjectDice(bw,bwa);
        objJaccard(i-1,1) = ObjectJaccard(bw,bwa);
        score(i-1,1) = F1score_pixel(bw,bwa);        
        dice(i-1,1) = Dice(bw,bwa);
        jaccard(i-1,1) = Jaccard(bw,bwa);
    end
end

%  precision = TP./(TP + FP);
%  recall = TP./(TP + FN);
%
%  score = (2*precision.*recall)./(precision+recall);