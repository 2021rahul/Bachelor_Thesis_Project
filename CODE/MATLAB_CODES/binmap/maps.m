for i = 19:20
    file = strcat(int2str(i),'_tl.mat');
    load(file);
    S = zeros(size(bin_map));
    G = S;
    S = S | bin_map;
%     file = strcat(int2str(i),'_tr.mat');
%     load(file);
%     S = S | bin_map;
    file = strcat(int2str(i),'_bl.mat');
    load(file);
    S = S | bin_map;
%     file = strcat(int2str(i),'_br.mat');
%     load(file);
%     S = S | bin_map;
%     figure;
%     imshow(S)
    file = strcat(int2str(i),'.bmp');
    imwrite(S, file)
    
    file = strcat(int2str(i),'_tlbin.mat');
    load(file);
    G = G | bin_map;
%     file = strcat(int2str(i),'_trbin.mat');
%     load(file);
%     G = G | bin_map;
    file = strcat(int2str(i),'_blbin.mat');
    load(file);
    G = G | bin_map;
%     file = strcat(int2str(i),'_brbin.mat');
%     load(file);
%     G = G | bin_map;
%     figure;
%     imshow(G)
    file = strcat(int2str(i),'_bin.bmp');
    imwrite(G, file)
end