for i=0:12
    nimg = strcat(int2str(i),'.mat');
    load(nimg);
    img = bin_map.*255;
    img = uint8(img);
    newimg = img(26:415,26:415);
    imshow(newimg)
    imwrite(newimg,strcat(int2str(i),'.jpg'));
end