clc
clear

load('../MOS.mat')
addpath('./YUVtoolbox')
databasePath = 'D:/DATA/LIVE-SJTU_AVQA/';
frameSkip = 2;
patchSize = 224;
position_width = [];
position_height = [];

for h = 1:200:resolution(1)
    if h < resolution(1) - patchSize
        for w = 1: 200: resolution(2)
            if w < resolution(2) - patchSize + 1
                position_width = [position_width, w];
                position_height = [position_height, h];
            else
                position_height = [position_height, h];
                position_width = [position_width, resolution(2) - patchSize + 1];
                break
            end
        end
    else
        for w = 1: 200: resolution(2)
            if w < resolution(2) - patchSize + 1
                position_height = [position_height, resolution(1) - patchSize + 1];
                position_width = [position_width, w];
            else
                position_height = [position_height, resolution(1) - patchSize + 1];
                position_width = [position_width, resolution(2) - patchSize + 1];
                break
            end
        end
        break
    end
end
position = int16([position_height; position_width]);

sal_index = {};
i = 0;
for iVideo = 1:3:length(disNames)
    iVideo
    i = i + 1;
    disVideo = [databasePath 'Distorted/' disNames{iVideo,1}];
    
    disID = fopen(disVideo);
    sort_frame = [];
    
    for iframe = 1:192
        iframe
        [disY, disCb, disCr] = readframeyuv420(disID, resolution(1), resolution(2));
        if feof(disID)
            break
        end
        if mod(iframe,frameSkip)~=1
            continue
        end
        disY = reshape(disY, [resolution(2) resolution(1)])';
        disCb = reshape(disCb, [resolution(2)/2 resolution(1)/2])';
        disCr = reshape(disCr, [resolution(2)/2 resolution(1)/2])';
        
        disRGB = yuv2rgb(disY,disCb,disCr);
        sal_img = fes_index(disRGB);
        sal_img = imresize(sal_img,resolution(1)/size(sal_img,1),'bicubic');
        sal_sum = zeros(1,length(position));
        for iposition = 1:length(position)
            sal_sum(iposition) = sum(sum(sal_img(position(1,iposition):position(1,iposition)+patchSize-1, ...
            position(2,iposition):position(2,iposition)+patchSize-1)));
        end
        [sal_sum, sort_position] = sort(-sal_sum);
        sort_frame = [sort_frame;sort_position(1:25)];     
    end
    sal_index{i} = sort_frame;
end    
%save('./SJTU_position.mat','sal_index');