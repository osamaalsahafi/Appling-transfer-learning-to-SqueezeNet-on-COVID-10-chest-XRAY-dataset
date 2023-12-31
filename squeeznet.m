clear all 
close all 
clc

net = squeezenet;

%Read dataset
imds = imageDatastore('datasets\COVID-19_Radiography_Dataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%Divide the data into 70% training data and 30% validation data sets.
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net); 
%Get the number of classes
numClasses = numel(categories(imdsTrain.Labels));

% create a new convolution layer for the new modle
newConvLayer =  convolution2dLayer([1, 1],numClasses,'WeightLearnRateFactor',100,'BiasLearnRateFactor',100,"Name",'new_conv');
lgraph = replaceLayer(lgraph,'conv10',newConvLayer);

% create a new classificaton layer for the new modle
newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassificatonLayer);

%Resize dataset
augmenter = imageDataAugmenter('RandXReflection', true);
resizedimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);
resizedimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);

% Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',300, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',0.00001, ...
    'ValidationFrequency',100, ...
    'Verbose',false, ...
    'shuffle', 'every-epoch', ...
    'ValidationData',resizedimdsValidation, ...
    'Plots','training-progress');

netTransfer = trainNetwork(resizedimdsTrain,lgraph,options);

