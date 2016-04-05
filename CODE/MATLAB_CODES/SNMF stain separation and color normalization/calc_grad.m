% Calculate the gradient

function []=calc_grad(grayimg, method)

if strcmp(method,'prewitt')
    filt_x=[-1 0 1 ; -1 0 1 ; -1 0 1];
    filt_y=[-1 -1 -1 ; 0 0 0 ; 1 1 1];

elseif strcmp(method,'sobel')
    filt_x=[-1 0 1 ; -2 0 2 ; -1 0 1];
    filt_y=[-1 -2 -1 ; 0 0 0 ; 1 2 1];  
end
% horizontal gradient
    gradx=imfilter(grayimg./255,filt_x,'symmetric');
    grady=imfilter(grayimg./255,filt_y,'symmetric');
    grad_img=sqrt(double(gradx.^2+grady.^2));
    gradangle_img=atan2(double(grady),double(gradx));    
    new=max(gradx,grady)*255;
    figure;imshow(grad_img)
    figure;imshow(gradangle_img)
end
