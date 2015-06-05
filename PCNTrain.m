function [ftrain,model]=PCNTrain(outImg,Option)
% Version 1.000
%
% Code provided by Gan Yanhai, Liu Jun and Dong Junyu
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This function is used to train the network
% trainData is the input data to train the network,it consists of many
% different columns and each column is a vectorized input sample ex. an
% image.
% =======INPUT=============
% Option contains the parameters used to set up the network,following are the list of its components:
% Option.imgSize:size of input image
% Option.numStage:layers of the network
% Option.numFilters:the number of filters of every group in a layer
% Option.patchSize:size of patches derived from input image
% Option.patchStep:define the length to move when obtain patches from input
% image in two directions
% Option.pooling:declare whether to add a pooling layer after every
% convolution layer
% Option.poolingSize:size of window to make pooling
% Option.poolingMethos:define the method used to make pooling
% Option.unionType:organization of every layer
% =======OUTPUT============
% ftrain:the output features of trainData through the net
% mode:save the filters of ecah layer

sampleNum=size(outImg,2);
height=Option.imgSize(1);
width=Option.imgSize(2);
for i=1:Option.numStage
    num=Option.numFilters(i);
    if i==1
       V=GetFilters(outImg,[height width],Option.patchSize,Option.patchStep,num,Option.imgFormat); 
       model.V{i}=V;
       [outImg,height,width,Idx]=PCNOutput(outImg,[height width],Option.unionType{i},Option.patchSize,Option.patchStep,Option.numFilters(1),V,Option.pooling,Option.poolingSize,Option.poolingMethod,Option.imgFormat);
    else
        type=Option.unionType{i};
        group=size(type,2);
        tempImg=zeros(size(outImg,1),sampleNum);
        for j=1:group
            index=find(type(:,j));
            for k=1:length(index)
                tempImg=tempImg+outImg(:,Idx==index(k));
            end
            tempImg=tempImg/length(index);
            V=GetFilters(tempImg,[height width],Option.patchSize,Option.patchStep,Option.numFilters(i),'gray');
            model.V{i}{j}=V;
        end
        model.V{i}=[model.V{i}{:}];
        [outImg,height,width,Idx]=PCNOutput(outImg,[height width],Option.unionType{i},Option.patchSize,Option.patchStep,num,model.V{i},Option.pooling,Option.poolingSize,Option.poolingMethod,'gray');
    end
%     model.height(i)=height;
%     model.width(i)=width;

end
clear V;
span=num*group;

switch Option.extractMethod
    case 'gabor'
        ftrain=zeros(span*2,sampleNum);
        k=1;
        f=cell(span,1);
        for i=1:sampleNum
            for j=1:span
                m=mean(outImg(:,k));
                s=var(outImg(:,k));
                f{j}=[m,s];
                k=k+1;
            end
            ftrain(:,i)=[f{:}]';
        end
        clear f;
        clear outImg;
    case 'hashHist'
        idx=1:sampleNum;
        idx=kron(idx,ones(1,span));
        inImg=mat2imgcell(outImg,height,width,'gray');
        clear outImg;
        [ftrain,~] = HashingHist(Option,idx,inImg);
    otherwise
        height=size(outImg,1);
        height=height*span;
        ftrain=zeros(height,sampleNum);
        for i=1:sampleNum
            ftrain(:,i)=reshape(outImg(:,(i-1)*span+1:i*span),height,1);
        end
        clear outImg;
end
end

