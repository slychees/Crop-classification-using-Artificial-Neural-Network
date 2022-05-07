%% Crop Classification using deep learning 

%% Get and organize the data
% We will use a subset of the MIT Places dataset: http://places2.csail.mit.edu/download.html 

%% Create image data store
imds = imageDatastore(fullfile('Crop'),...
'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');

%% Count number of images per label 
labelCount = countEachLabel(imds);

%% Create training and validation sets
[imdsTraining, imdsValidation, imdsTesting] = splitEachLabel(imds, 0.6, 0.3, 'randomized');


%resizing trainingSet to size 224X224
I = read(imdsTraining); % % read first image from imds
% figure,
% subplot(121); imshow(I); title('First image, before resize'); axis on;
% % -----------------------------------------------------
% % Now from this point, use the custom reader function
imdsTraining.ReadFcn = @customreader;
% % Reset the datastore to the state where no data has been read from it.
reset(imdsTraining);
J = read(imdsTraining); % % read the first image again (because we reset read)
%subplot(122); imshow(J); title('First image, after resize'); axis on;
K = read(imdsTraining); % % read the second image
L = read(imdsTraining); % % read the third image


%% Use image data augmentation to handle the resizing 
% The original images are 256-by-256. The input layer of the CNNs used in
% this example expects them to be 224-by-224.
inputSize = [224,224,3];
augimdsTraining = augmentedImageDatastore(inputSize(1:2),imdsTraining);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
augimdsTesting = augmentedImageDatastore(inputSize(1:2),imdsTesting);

%% PART 1: Use simple CNN built from scratch

%% Define Layers
layers = [
    imageInputLayer([224 224 3])
    convolution2dLayer(3,16,'Padding',1) %3 kernel size
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,64,'Padding',1)
    %batchNormalizationLayer
    %reluLayer
    %maxPooling2dLayer(2,'Stride',2)
    %dropoutLayer %
    %crossChannelNormalizationLayer(10) %
    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];

%% Specify Training Options
options = trainingOptions('sgdm',...
    'MaxEpochs',30, ...
    'ValidationData',augimdsValidation,...
    'ValidationFrequency',1,...
    'InitialLearnRate', 0.0003,...
    'Verbose',true,...
    'Plots','training-progress');

%% Train network
baselineCNN = trainNetwork(augimdsTraining,layers,options);

%% (OPTIONAL) Analyze network
analyzeNetwork(baselineCNN)
% You should get 0 warnings and 0 errors.



%% Classify and Compute Accuracy For Training
predictedLabelsT = classify(baselineCNN,augimdsTraining);
TLabels = imdsTraining.Labels;
trainingAccuracy = sum(predictedLabelsT == TLabels)/numel(TLabels);



%% Classify and Compute Accuracy For Validation
predictedLabels = classify(baselineCNN,augimdsValidation);
valLabels = imdsValidation.Labels;
baselineCNNAccuracy = sum(predictedLabels == valLabels)/numel(valLabels);



%% Classify and Compute Accuracy For Testing
predictedLabelsTest = classify(baselineCNN,augimdsTesting);
TestLabels = imdsTesting.Labels;
testingAccuracy = sum(predictedLabelsTest == TestLabels)/numel(TestLabels);



%% (OPTIONAL) Plot confusion matrix for Validation set
figure, plotconfusion(valLabels,predictedLabels);
title('Validation set confMAT');

%% (OPTIONAL) Plot confusion matrix for Training set
figure, plotconfusion(TLabels,predictedLabelsT)
title('Training set confMAT');


%% (OPTIONAL) Plot confusion matrix for Testing set
figure, plotconfusion(TestLabels,predictedLabelsTest)
title('Testing set confMAT');


%comparing performances of train, validation sets

[trainingAccuracy,baselineCNNAccuracy,testingAccuracy]


figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,incorrect_idx(idx(i)));
    imshow(I)
    label = YPred(incorrect_idx(idx(i)));
    GT = YValidation(incorrect_idx(idx(i)));
    title("Predicted: " + string(label) + " | Correct: " + string(GT));
end





%function for resizing trainingSet images
function data = customreader(filename)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
data = data(:,:,min(1:3, end));
data = imresize(data, [224 224]);
end

%function for resizing testSet images
function data = customreader1(filename)
onState = warning('off', 'backtrace');
c = onCleanup(@() warning(onState));
data = imread(filename);
data = data(:,:,min(1:3, end)); 
data = imresize(data, [224 224]);
end
