clc;
clear;
close all;

%-----Choosing which eight i'd like to see --------
%-----(eight = 1 --> means the 1st eight etc.)-----
% Change also the line 32
eight = 1;
N = 500;

%Loading Data
data21 = load ('data21.mat');
A1 = data21.A_1; A2 = data21.A_2;
B1 = data21.B_1; B2 = data21.B_2;
data22 = load ('data22.mat');
Xi = data22.X_i;
Xn = data22.X_n;

%Initial Conditions
gradJ = [];
iterations = 2000;
learning_rate = 0.05;
T = eye(N,784);
Xn_new = T*Xn;
Xn_new(N:784,:) = 0;

% Z1 = load("Z1.mat");
% Z2 = load("Z2.mat");
% Z3 = load("Z3.mat");
% Z4 = load("Z4.mat");

% Zintilial = Z1.Zall(:,i); %Input from file Z.Zall(:,i) /{i = 1,2,3,4,5,6,7}/ 
                          % where i choose the best initial Z for the algorithm

Zintilial = randn(10,1);

lamb = 1;       %ADAM constant
P = 0;          %ADAM Power
c = 10^(-6);    %ADAM small number
Z = Zintilial;  %Input

%Gradient Descent
for i=1:iterations        

    W1 = A1*Z + B1;
    Z1 = max(W1,0); %ReLU
    W2 = A2*Z1 + B2;
    X = 1./(1+exp(W2)); %Sigmoid

    %Steps for GRAD_Z(PHI(X)) = U_0
    f1_grad_W1 = DerivativeReLU(W1);
    f2_grad_W2 = -exp(W2)./((1 + exp(W2)).^2);

    U_2 = DerivativePhi(X , T, Xn, N, eight);
    V_2 = U_2 .* f2_grad_W2;
    U_1 = A2' * V_2;
    V_1 = U_1 .* f1_grad_W1;
    U_0 = A1' * V_1;
    gradJ = N*U_0 + 2*Z;
    
    %ADAM Power for Normalization
    P = (1-lamb)*P + lamb * gradJ.^2;
    lamb = 0.001;
    
    %Algorithm
    Z_next = Z - learning_rate * gradJ./sqrt(c + P);
    Z = Z_next;  

end

figure(1)
% ---Xi---
subplot(1,3,1)
imshow(reshape(Xi(:,eight),28,28));
% ---Xn---
subplot(1,3,2)
imshow(reshape(Xn_new(:,eight),28,28));
% ---New---
subplot(1,3,3)
imshow(reshape(X,28,28));


function y = DerivativeReLU(W)
    W(W(:,:)<=0) = 0;
    W(W(:,:)>0) = 1;
    y = W;
end

function y = DerivativePhi(X , T, Xn, N, eight)
    y = (2 / (norm( T*X - Xn(1:N,eight) )^2 ) ) * T' * (T*X - Xn(1:N,eight));
end