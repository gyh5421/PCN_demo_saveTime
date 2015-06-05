function V = GetFilters(inImg,imgSize,patchSize,patchStep,numFilters,imgFormat)


addpath('./Utils')

num = size(inImg,2);
RandIdx=randperm(num);

inImg=mat2imgcell(inImg,imgSize(1),imgSize(2),imgFormat);
Rx = zeros(patchSize(1)*patchSize(2)*size(inImg{1},3),patchSize(1)*patchSize(2)*size(inImg{1},3));

for i = RandIdx 
    im = im2col_general(inImg{i},patchSize,patchStep); % collect all the patches of the ith image in a matrix
    im = bsxfun(@minus, im, mean(im)); % patch-mean removal 
    Rx = Rx + im*im'; % sum of all the input images' covariance matrix
end
Rx = Rx/(num*size(im,2));
[E,D] = eig(Rx);
[~,ind] = sort(diag(D),'descend');
V = E(:,ind(1:numFilters));  % principal eigenvectors 
clear Rx;



 



