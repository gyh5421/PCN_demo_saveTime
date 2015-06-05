function ftest=PCNExtFea(testData,model,Option)
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

% This function is used to extract features through PCN,it can process one
% sample a time
% Input:
% ftest is the input image,model contains the structure of PCN trained
% before and option contains some hyper-parameters
% Output:the features

height=Option.imgSize(1);
width=Option.imgSize(2);
ftest=testData;
clear testData;
for i=1:Option.numStage
    [ftest,height,width,~]=PCNOutput(ftest,[height,width],Option.unionType{i},Option.patchSize,Option.patchStep,Option.numFilters(i),model.V{i},Option.pooling,Option.poolingSize,Option.poolingMethod,Option.imgFormat);
end

num=size(ftest,2);
switch Option.extractMethod
    case 'gabor'
        f=zeros(num*2,1);
        k=1;
        for i=1:num
            m=mean(ftest(:,i));
            s=var(ftest(:,i));
            f([k,k+1])=[m;s];
            k=k+2;
        end
        ftest=f;
        clear f;
    case 'hashHist'
        inImg=mat2imgcell(ftest,height,width,'gray');
        [ftest,~] = HashingHist(Option,ones(1,num),inImg);
    otherwise
        ftest=ftest(:);
end
end

