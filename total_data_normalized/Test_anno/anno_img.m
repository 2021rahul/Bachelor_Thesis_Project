cd('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/data_normalized/Test_anno');
% cd ('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/data_normalized/Test');
dir_list = dir;
for i=3:15
%     cd ('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/data_normalized/Test');
    cd('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/data_normalized/Test_anno')
    name = dir_list(i).name;
    namepart = strtok(name,'.');
    newname = strcat(namepart,'_bin.jpg');
    im = imread(name);
    newim = imrotate(im,-90);
    newim = fliplr(newim);  %for greyscale image
%     for i=1:3
%         newim(:,:,i) = fliplr(newim(:,:,i));
%     end
    newim = newim(26:390+25,26:390+25,:);
    cd ('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/data_normalized/result_maps');
    imwrite(newim,newname);
end