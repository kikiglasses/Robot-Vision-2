unzip('CIFAR-10.zip');
imdsTrain = imageDatastore('CIFAR-10/train/', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imdsValidation = imageDatastore('CIFAR-10/test/', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%
net = vgg16;
%%
analyzeNetwork(net);

%%
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);

learnableLayer = net.Layers(39);
% midLayer = net.Layers(40);
classLayer = net.Layers(41);

numClasses = numel(categories(imdsTrain.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
            'Name','new_fc', ...
            'WeightLearnRateFactor',10, ...
            'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% newsoftmaxLayer = softmaxLayer('Name','new_softmax');
% lgraph = replaceLayer(lgraph,midLayer.Name,newsoftmaxLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph);
% ylim([0,10]);

%%
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%% Augment training and test data to correct size (224x224)

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',0.001, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%% Train the network
net = trainNetwork(augimdsTrain,lgraph,options);

%% Accuracy and Cross-Entropy Calculated
[YPred,probs] = classify(net,augimdsValidation);

accuracy = mean(YPred == imdsValidation.Labels)

entropy=0;
for i = 1:size(YPred)
    p = 0;
    for j = 1:10
        if probs(i,j) > p
            p = probs(i,j);
        end
    end
    for j = 1:10
        entropy = entropy + log(p);
    end
end
entropy = -1*entropy/size(YPred,1)

correct_count = zeros([1,numel(categories(imdsTrain.Labels))]);
for i = 1:numel(categories(imdsTrain.Labels))
    a = categories(imdsTrain.Labels);
    correct_count(i) = sum(YPred == a(i) & YPred == imdsValidation.Labels);
end
correct_count
%%
worst = 'cat'

%% Show 4 examples of the network classifying
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

            %%       Extend network for bicycles        %%

unzip('CIFAR-10.zip');
unzip('CIFAR-100.zip');

imdsTrain = imageDatastore({'CIFAR-10/train/','CIFAR-100/train/'}, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
imdsValidation = imageDatastore({'CIFAR-10/test/','CIFAR-100/test/'}, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%
net = vgg16;
%%
analyzeNetwork(net);

%%
inputSize = net.Layers(1).InputSize;
lgraph = layerGraph(net);

learnableLayer = net.Layers(39);
% midLayer = net.Layers(40);
classLayer = net.Layers(41);

numClasses = numel(categories(imdsTrain.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
            'Name','new_fc', ...
            'WeightLearnRateFactor',10, ...
            'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% newsoftmaxLayer = softmaxLayer('Name','new_softmax');
% lgraph = replaceLayer(lgraph,midLayer.Name,newsoftmaxLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
% plot(lgraph);
% ylim([0,10]);

%%
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%% Augment training and test data to correct size (224x224)

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',0.001, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%% Train the network
net = trainNetwork(augimdsTrain,lgraph,options);

%% Accuracy and Cross-Entropy Calculated
[YPred,probs] = classify(net,augimdsValidation);

accuracy = mean(YPred == imdsValidation.Labels)

entropy=0;
for i = 1:size(YPred)
    p = 0;
    for j = 1:11
        if probs(i,j) > p
            p = probs(i,j);
        end
    end
    for j = 1:11
        entropy = entropy + log(p);
    end
end
entropy = -1*entropy/size(YPred,1)


correct_count = zeros([1,numel(categories(imdsTrain.Labels))]);
for i = 1:numel(categories(imdsTrain.Labels))
    a = categories(imdsTrain.Labels);
    correct_count(i) = sum(YPred == a(i) & YPred == imdsValidation.Labels);
end
correct_count
%%
worst = 'cat'

%% Show 4 examples of the network classifying
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%% Classify Stable Diffusion image
SF = imresize(imread("SF.jpg"),[224,224]);
[predicted_result,prob] = classify(net,SF);
prob_bicycle = prob(3);
prob_horse = prob(9);

