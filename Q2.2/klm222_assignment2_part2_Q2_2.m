%% Load data
filenameImagesTrain = 'MNIST\train-images-idx3-ubyte';
filenameLabelsTrain = 'MNIST\train-labels-idx1-ubyte';
filenameImagesTest = 'MNIST\t10k-images-idx3-ubyte';
filenameLabelsTest = 'MNIST\t10k-labels-idx1-ubyte';

XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);

XTest = processImagesMNIST(filenameImagesTest);
YTest = processLabelsMNIST(filenameLabelsTest);

%% Random seeds
idx1 = randperm(128,24);
idx2 = randperm(128,6);

XTrain_ex = XTrain(:,:,:,idx1);
YTrain_ex = YTrain(idx1);

XTest_ex = XTest(:,:,:,idx2);
YTest_ex = YTest(idx2);

%% Show examples
figure;

trainCount = zeros([1,numel(categories(YTrain))]);
testCount = zeros([1,numel(categories(YTest))]);

for i = 1:24
    subplot(5,6,i);
    imshow(XTrain_ex(:,:,:,i));
    label = str2double(string(YTrain_ex(i)));
    trainCount(label+1) = trainCount(label+1) + 1;
end

for i = 25:30
    subplot(5,6,i);
    imshow(XTest_ex(:,:,:,i-24));
    label = str2double(string(YTest_ex(i-24)));
    testCount(label+1) = testCount(label+1) + 1;
end

trainCount
testCount

%% Augment the images
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20,30], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-3,2], ...
    'RandYTranslation', [-2,4] ...
    );

augimdsTrain = augmentedImageDatastore([256,256,3],XTrain_ex,YTrain_ex, ...
    'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);

%% Show augmented images
figure;
reset(augimdsTrain);
ims = read(augimdsTrain);
for i = 1:24
    subplot(4,6,i);
    I = cell2mat(ims{i,1});
    imshow(I);
end

%% New Parity label arrays

odd = ['1';'3';'5';'7';'9'];

YPTrain = ismember(string(YTrain_ex),odd);
YPTest = ismember(string(YTest_ex),odd);

YPTrain = categorical(YPTrain);
YPTest = categorical(YPTest);

augimdsTrain = augmentedImageDatastore([256,256,3],XTrain_ex,YPTrain, ...
    'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore([256,256,3],XTest_ex,YPTest, ...
    'ColorPreprocessing','gray2rgb');

%% Set up the network

% Borrow some network weights from darknet19
darknet = darknet19;

layers = [...
    imageInputLayer([256 256 3])
    darknet.Layers(2)
    darknet.Layers(3)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    darknet.Layers(6)
    darknet.Layers(7)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    darknet.Layers(10)
    darknet.Layers(11)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    darknet.Layers(13)
    darknet.Layers(14)
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs',35, ...
    'InitialLearnRate',0.02, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',50, ...
    'ValidationPatience',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%% Train the network
net = trainNetwork(augimdsTrain,layers,options);
%%
analyzeNetwork(net);
%% Load custom test data

imdsCustomTest = imageDatastore('custom_test/', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

augimdsCustomTest = augmentedImageDatastore([256,256,3], imdsCustomTest, ...
    "ColorPreprocessing","gray2rgb");

[YPred,probs] = classify(net,augimdsCustomTest);

