 function im3 = fes_index(im1)
sr  = 2;
mr  = 1;
len = 3;
par = 2;
gll = 5;
gss = 10;
aaa = 54;
bbb = 255;
www = 3;
sr2 = 2;
mr2 = 0.2;
ccc = 0.3;
%% pre-processing
im2 = double(imresize(im1,aaa/size(im1,1),'bicubic'));
%% residual estimation
ar = zeros(size(im2));
for i = 1:3
ar(:,:,i) = AR_saliency(im2(:,:,i),sr,mr);
end
bi = bfltColor(im2/255,www,sr2,mr2)*255;
ar = (ar+ccc*bi)/(1+ccc);
%% color transfer
labT = makecform('srgb2lab');
ar = double(applycform(uint8(ar),labT));
im2 = double(applycform(uint8(im2),labT));
%% saliency detection
im3 = zeros(size(im2),'gpuArray');
for j = 1:3
    ar1 = ar(:,:,j)-im2(:,:,j);
    ar2 = round(padarray(ar1,[len len],'symmetric'));
    ar3 = im2col(ar2,[len*2+1 len*2+1],'sliding');
    ar4 = zeros(size(ar3,2),1);
        for i = 1:size(ar3,2)
            nn=hist(ar3(:,i),-255:255);
            p=(1+bbb*nn)/sum(1+bbb*nn);
            ar4(i) = -sum(p.*log2(p));
        end
    ar5 = reshape(ar4,[size(im2,1) size(im2,2)]);
    ar6 = imfilter(ar5,fspecial('gaussian',gll,gss),'symmetric','conv');
    ar6 = mat2gray(ar6).^par;
    ar7 = imresize(ar6,size(im2(:,:,1)),'bicubic');
    im3(:,:,j) = ar7;
end
im3 = (1.5*im3(:,:,1)+im3(:,:,2)+im3(:,:,3))/(2+1.5);
%%
function B = bfltColor(A,w,sigma_d,sigma_r)
% Convert input sRGB image to CIELab color space.
A = applycform(A,makecform('srgb2lab'));
% Pre-compute Gaussian domain weights.
[X,Y] = meshgrid(-w:w,-w:w);
G = exp(-(X.^2+Y.^2)/(2*sigma_d^2));
% Rescale range variance (using maximum luminance).
sigma_r = 100*sigma_r;
% Apply bilateral filter.
dim = size(A);
B = zeros(dim);
for i = 1:dim(1)
for j = 1:dim(2)
% Extract local region.
iMin = max(i-w,1);
iMax = min(i+w,dim(1));
jMin = max(j-w,1);
jMax = min(j+w,dim(2));
I = A(iMin:iMax,jMin:jMax,:);
% Compute Gaussian range weights.
dL = I(:,:,1)-A(i,j,1);
da = I(:,:,2)-A(i,j,2);
db = I(:,:,3)-A(i,j,3);
H = exp(-(dL.^2+da.^2+db.^2)/(2*sigma_r^2));
% Calculate bilateral filter response.
F = H.*G((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);
norm_F = sum(F(:));
B(i,j,1) = sum(sum(F.*I(:,:,1)))/norm_F;
B(i,j,2) = sum(sum(F.*I(:,:,2)))/norm_F;
B(i,j,3) = sum(sum(F.*I(:,:,3)))/norm_F;
end
end
B = applycform(B,makecform('lab2srgb'));
%%
function imgrec = AR_saliency(imgin,sr,mr)
imgt=padarray(imgin,[sr+mr sr+mr],'symmetric');
imgrec=zeros(size(imgin));
[m n]=size(imgt);
N=(2*sr+1)^2-1;
K=(2*mr+1)^2-1;
A=zeros(N,K+1);
for ii=mr+sr+1:m-sr-mr
for jj=mr+sr+1:n-sr-mr
con=1;
patch0=imgt(ii-mr:ii+mr,jj-mr:jj+mr);
for iii=-sr:+sr
for jjj=-sr:+sr
if iii==0&&jjj==0
continue;
end
patch=imgt(ii+iii-mr:ii+iii+mr,jj+jjj-mr:jj+jjj+mr);
vec=patch(:);
A(con,:)=vec';
con=con+1;
end
end
b=A(:,mr*(2*mr+2)+1);
A2=A;
A2(:,mr*(2*mr+2)+1)=[];
if rcond(A2'*A2)<1e-7
a = ones(K,1)/K;
else
a = A2\b; %解方程 A2=ba
end
vec0=patch0(:);
vec0(mr*(2*mr+2)+1)=[];
rec=vec0'*a;
imgrec(ii-sr-mr,jj-sr-mr)=rec;
end
end