% Nuclei segmentation using SNMF

% Parameters
nstains=2;    % number of stains
lambda=0.1;   % default value sparsity regularization parameter, 
% lambda=0 equivalent to NMF
tic;
% Stain separation (V=WH)
[Wi, Hi,Hiv,stains]=stainsep(I,nstains,lambda);
time=toc
% Visuals (for 2 stains)
figure;
subplot(131);imshow(I);xlabel('Input')
subplot(132);imshow(stains{1});xlabel('stain1')
subplot(133);imshow(stains{2});xlabel('stain2')

% Save images of useful components for segmentation
imagename='ytma10_010704_benign1_ccd'
savefolder='D:\Dropbox\MyPhDThesis\Chapter3_SNMF_nucleiseg\'
% imwrite(Hi(:,:,1),[imagename,'_','DenMap','.png'])
% imwrite(imcomplement(rgb2gray(stains{1})),[imagename,'_','graystain','.png'])
% imwrite(stains{1},[imagename,'_','RGBstain1','.png'])
% imwrite(stains{2},[imagename,'_','RGBstain2','.png'])

gray_H=imcomplement(rgb2gray(stains{1}));
DenMap_H=Hi(:,:,1);

% Use above maps for segmentation

[grad_image]=calc_grad(gray_H,'prewitt');
