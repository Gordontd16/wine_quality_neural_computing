%% ************************************************************************
%                       EXPLORATORY DATA ANALYSIS
% *************************************************************************

% AIM: Calculate summary statistics for features; explore variable
% distributions and skew; visualise class imbalance; generate a correlation 
% matrix in order to visualise correlations between features; create a
% parallel coordinates plot.

% Clear workspace and Command window
clear; clc; close all;

% Load the dataset
data = readtable('winequality-white.csv', 'PreserveVariableNames', true);

% Calculate 80th percentile or "quality" dimension
perc = prctile(data.quality,80);
fprintf('The 80th percentile is equivalent to a quality score of %d.\n\n', perc)
% The answer is 7

% Define a new variable 'good_quality' for wines with quality >= 7.
data.good_quality = data.quality >= 7;
% Make "good_quality" a categorical variable
data.good_quality = categorical(data.good_quality);

% Data Shape
[m, n] = size(data); % m = number of rows ; n = number of columns

% Print Table Summary
fprintf('------------------------------------------------------------------\n')
fprintf('The dataset has %d Rows and %d Columns.\n\n', m, n)
fprintf('------------------------------------------------------------------\n')
statarray = grpstats(data(:, 1:12),[], {'mean','std'});

%% Figure 1: Univariate Analysis - Histograms of Descriptors + Histogram of Target
figure('pos',[10 10 1400 800])
for col_index = 1:n-1
    subplot(4,3,col_index)
    histogram(data{:, col_index})
    title(sprintf('Histogram of %s', data.Properties.VariableNames{col_index}))
end
skew = skewness(data{:,1:col_index});
k = kurtosis(data{:,1:col_index});

%% Figure 2: Histogram of target
figure('pos',[10 1000 500 400])
histogram(data.good_quality)
xlabel('Wine Quality >7'); ylabel('Number of Wines');

%% Figure 3: Bivariate Analysis - Correlation
    % Compute Correlation Matrix (Pearson coefficient)
corrMatrix = corr(table2array(data(:, 1:col_index)),'type','Pearson');
    % Plot the Correlation Matrix
figure('pos',[1000 1000 500 400])
labels1 = data.Properties.VariableNames(1:col_index); % Numeric attributes' names
colormap('hot') % Color Theme
clims = [-1, 1]; % Color Scale limits
imagesc(corrMatrix, clims) % Visualise
colorbar % Show colorbar
    % Add Text (values) to Matrix
textStrings = num2str(corrMatrix(:), '%0.2f'); 
textStrings = strtrim(cellstr(textStrings));
[x, y] = meshgrid(1:12);  % Create x and y coordinates for the strings on the matrix
hStrings = text(x(:), y(:), textStrings(:), 'HorizontalAlignment', 'center');
set(hStrings, 'color', 'black') % Set color of text
    % Set Axis Labels and Title
set(gca, 'XTickLabel', labels1, 'XTick', 1:col_index, 'XTickLabelRotation',45); % set x-axis labels
set(gca, 'YTickLabel', labels1, 'YTick', 1:col_index, 'YTickLabelRotation',45); % set y-axis labels
%title('Correlation Matrix')

%% Figure 4: Multivariate Analysis - Parallel Coordinates
    % Scaling Numeric Columns
dataNum = table2array(data(:, 1:col_index-1)); % Table to Matrix
dataMean = mean(dataNum); dataStd = std(dataNum);
dataNumNorm = (dataNum - dataMean)./dataStd; % Normalise
    % Visualise using Parallel Coordinates Grouped by Target
figure('pos',[450 10 500 400])
labels2 = data.Properties.VariableNames(1:col_index-1);
p=parallelcoords(dataNumNorm, 'group', data.good_quality, 'labels', labels2, 'quantile', 0.25);
xtickangle(45);
