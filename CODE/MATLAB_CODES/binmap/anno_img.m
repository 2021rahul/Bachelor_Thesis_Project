%cd('/media/rahul/1098D6BA98D69E12/academics/SEMESTER_7/BTP/data/test_set_anno');
cd ('E:\academics\SEMESTER_7\BTP\data\test_set_anno');
dir_list = dir;
for i=3:14
    cd ('E:\academics\SEMESTER_7\BTP\data\test_set_anno');
    name = dir_list(i).name;
    namepart = strtok(name,'.');
    newname = strcat(namepart,'_bin.jpg');
    im = imread(name);
    newim = imrotate(im,-90);
    newim = fliplr(newim);  %for greyscale image
%     for i=1:3
%         newim(:,:,i) = fliplr(newim(:,:,i));
%     end
    newim = newim(16:495,16:495,:);
    cd ('E:\academics\SEMESTER_7\BTP\data\result_maps');
    imwrite(newim,newname);
end