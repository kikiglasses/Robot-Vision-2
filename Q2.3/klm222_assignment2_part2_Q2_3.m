%% Load layers.mat
load layers.mat

%%
% Images don't have colour channel dimension
training_images = reshape(training_images,[96,96,1,100]);
test_images = reshape(test_images,[96,96,1,25]);

% Rename label for training data
training_labels = labels;

%% Network options

% Set training options
options = trainingOptions('adam', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',100, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule',"piecewise", ...
    'L2Regularization',0.001, ...
    'LearnRateDropPeriod',3, ...
    'LearnRateDropFactor',0.1, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%% Set up network architecture

layers2 = [...
    imageInputLayer([96 96])
    layers(2)
    layers(3)
    layers(4)
    layers(5)
    layers(6)
    layers(7)
    layers(8)
    layers(9)
    layers(10)
    layers(11)
    layers(12)
    layers(13)
    layers(14)
    layers(15)
    layers(16)
    % layers(17)
    % layers(18)
    fullyConnectedLayer(30)
    % layers(20)
    regressionLayer
    ];
%%
net = trainNetwork(training_images,training_labels,layers2,options);

%%
YPred = predict(net,test_images);

%% Calculate MSE
diff = YPred - test_labels;
error = zeros([1,25]);
best = 1;
for i = 1:25
    error(i) = 0;
    for j = 1:15
        error(i) = error(i) + sqrt( diff(i,2*j)^2 + diff(i,2*j-1)^2 );
    end
    if error(i) <= error(best)
        best = i;
    end
end
error = error/15;
%% Show figures
figure;
hold on;
imshow(test_images(:,:,1,best));
for j = 1:15
    hold on;
    plot( YPred(best,2*j), YPred(best,2*j-1), 'r+', 'MarkerSize', 6, 'LineWidth', 2 );
    plot ( test_labels(best,2*j), test_labels(best,2*j-1), 'g+', 'MarkerSize', 6, 'LineWidth', 2)
end


