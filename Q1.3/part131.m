%% Load image

I = im2double(im2gray(imread("aston_webb.jpg")));

%%
figure;

subplot(3,2,1);
hold on;
imshow(I);
points = detectMinEigenFeatures(I);
plot(points.selectStrongest(200));
title("Minimum Eigenvalue");


subplot(3,2,2);
hold on;
imshow(I);
points = detectSURFFeatures(I);
plot(points.selectStrongest(200));
title("SURF");


subplot(3,2,3);
hold on;
imshow(I);
points = detectKAZEFeatures(I);
plot(points.selectStrongest(200));
title("KAZE");


subplot(3,2,4);
hold on;
imshow(I);
points = detectFASTFeatures(I);
plot(points.selectStrongest(200));
title("FAST");


subplot(3,2,5);
hold on;
imshow(I);
points = detectORBFeatures(I);
plot(points.selectStrongest(200));
title("ORB");


subplot(3,2,6);
hold on;
imshow(I);
points = detectHarrisFeatures(I);
plot(points.selectStrongest(200));
title("Harris-Stephens");


%%
minEigen = imread('minEigen.png');
SURF = imread('SURF.png');
KAZE = imread('KAZE.png');
FAST = imread('FAST.png');
ORB = imread('ORB.png');
Harris = imread('Harris.png');
montage({minEigen,SURF,KAZE,FAST,ORB,Harris},'BackgroundColor','white');



