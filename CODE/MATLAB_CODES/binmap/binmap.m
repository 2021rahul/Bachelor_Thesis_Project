for i=0:11
    nimg = strcat(int2str(i),'.mat');
    load(nimg);
    img = bin_map.*255;
    img = uint8(img);
    newimg = img(16:495,16:495);
    imshow(newimg)
    imwrite(newimg,strcat(int2str(i),'.jpg'));
end