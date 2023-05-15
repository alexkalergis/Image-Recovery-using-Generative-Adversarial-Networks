clc;
clear;
close all;

data21 = load ('data21.mat');
A1 = data21.A_1; A2 = data21.A_2;
B1 = data21.B_1; B2 = data21.B_2;

for i=1:100
    
    Z = randn(10,1); %Random Input
    
    W1 = A1*Z + B1;
    Z1 = max(W1,0); %ReLU
    W2 = A2*Z1 + B2;
    X = 1./(1+exp(W2)); %Sigmoid
    
    subplot(10,10,i)
    imshow(reshape(X,28,28))
end

