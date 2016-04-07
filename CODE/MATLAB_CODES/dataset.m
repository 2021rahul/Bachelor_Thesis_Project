for i=1:60
    img = strcat('testA_',int2str(i));
    img = strcat(img, '_bin.bmp');
    A = imread(img);   
    img = strcat('C:\Users\SURENDRA JAIN\Documents\AJ\Acads\VII sem\Bachelor_Thesis_Project\Exp1_a\Test_anno\', img);
    imwrite(A, img);
end

for i=1:43
    img = strcat(Grade(i),'_norm.bmp');
    img = char(img);
    A = imread(img);
    img = strcat('C:\Users\SURENDRA JAIN\Documents\AJ\Acads\VII sem\Bachelor_Thesis_Project\Exp1_c\Test\', img);
    imwrite(A,img);
end
