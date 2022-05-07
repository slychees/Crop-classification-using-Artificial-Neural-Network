%% PART 1: Baseline Classifier
%% Load image data
imds = imageDatastore('Crop', ...
    'IncludeSubfolders',true, 'FileExtensions','.png', ...
    'LabelSource','foldernames');

%% Create training and validation sets
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.6,0.3,0.1);



%% *%resizing trainingSet to size 224X224*
 
I = read(imdsTrain); % % read first image from imds
% figure,
% subplot(121); imshow(I); title('First image, before resize'); axis on;
% % -----------------------------------------------------
% % Now from this point, use the custom reader function
imdsTrain.ReadFcn = @customreader;
% % Reset the datastore to the state where no data has been read from it.
reset(imdsTrain);
J = read(imdsTrain); % % read the first image again (because we reset read)
%subplot(122); imshow(J); title('First image, after resize'); axis on;
K = read(imdsTrain); % % read the second image
L = read(imdsTrain); % % read the third image



numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

net = alexnet;
net.Layers

inputSize = net.Layers(1).InputSize;

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

%% Define Layers
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

%% Specify Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Train network
netTransfer = trainNetwork(augimdsTrain,layers,options);

%% (OPTIONAL) Analyze network
analyzeNetwork(netTransfer)
% You should get 0 warnings and 0 errors.

%% Classify and Compute Accuracy
[YPred,scores] = classify(netTransfer,augimdsValidation);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%% Classify and Compute Accuracy For Training
predictedLabelsT = classify(netTransfer,augimdsTrain);
TLabels = imdsTrain.Labels;
trainAccuracy = sum(predictedLabelsT == TLabels)/numel(TLabels);


%% Classify and Compute Accuracy For Validation
predictedLabelsV = classify(netTransfer,augimdsValidation);
valLabels = imdsValidation.Labels;
validationAccuracy = sum(predictedLabelsV == valLabels)/numel(valLabels);


%% Classify and Compute Accuracy For Testing
predictedLabelsTest = classify(netTransfer,augimdsTest);
TestLabels = imdsTest.Labels;
testingAccuracy = sum(predictedLabelsTest == TestLabels)/numel(TestLabels);


%% %% Compute accuracy and plot  Plot confusion matrix
figure, plotconfusion(valLabels,predictedLabelsV)

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

%% %% Display Train accuracy and Validation Accuracy
[trainAccuracy,validationAccuracy,testingAccuracy]





%% %% function for resizing trainingSet images
function data = customreader(filename)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
data = data(:,:,min(1:3, end));
data = imresize(data, [224 224]);
end



