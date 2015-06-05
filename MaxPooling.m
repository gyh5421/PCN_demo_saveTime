function [outImgs] = MaxPooling(images,imgSize,poolingSize,method)
num=size(images,2);
imgs=mat2imgcell(images,imgSize(1),imgSize(2),'gray');
clear images;
outImgs=cell(num,1);
for i=1:num
    im=imgs{i};
    patches=im2col_general(im,poolingSize,poolingSize);
    imgs{i}=[];
    if strcmp(method,'maxpooling')
        patches=max(patches);
    else
        patches=mean(patches,1);
    end
    outImgs{i}=patches';
end
outImgs=[outImgs{:}];
end

