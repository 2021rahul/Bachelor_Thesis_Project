clear;
close all
for i=0:12
    img = strcat(int2str(i),'.jpg');
    oimg = strcat(int2str(i),'_postnew.jpg');
    binary_image=imread(img);
    binary_image=im2bw(binary_image);
    se = strel('disk', 4, 0);
    image2=imopen(binary_image,se);
    %figure;imshow(binary_image);
    %figure;imshow(image2)
    image3=imfill(image2,'holes');
    %figure;imshow(image3)
    imwrite(image3, oimg);
end