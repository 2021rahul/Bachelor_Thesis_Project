clear all;
close all;
a=[0:79];
a = a(abs(a - 12) > eps(100));
a = a(abs(a - 76) > eps(100));
for i=1:numel(a)
    img = strcat(int2str(a(i)),'.bmp');
    aimg = strcat(int2str(a(i)),'_bin.bmp');
    oimg = strcat(int2str(a(i)),'_post.bmp');
    binary_image=imread(img);
    binary_image=im2bw(binary_image);
    %figure;imshow(binary_image)
    %figure; imshow(aimg);
    se = strel('disk', 5, 0);
    %image2=imopen(binary_image,se);
    %figure;imshow(binary_image);
    %figure;imshow(image2)
    image2=imfill(binary_image,'holes');
    image3=imopen(image2,se);
    image3 = imfill(image3,'holes');
    %figure;imshow(image3)
    imwrite(image3, oimg);    
end