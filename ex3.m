clc;
clear;
close all;

%-----Choosing which eight i'd like to see --------
%-----(eight = 1 --> means the 1st eight etc.)-----
% Change also the line 47
eight = 2;
N = 49;

%Loading Data
data21 = load ('data21.mat');
A1 = data21.A_1; A2 = data21.A_2;
B1 = data21.B_1; B2 = data21.B_2;
data23 = load ('data23.mat');
Xi = data23.X_i;
Xn = data23.X_n;

%Initial Conditions
gradJ = [];
iterations = 2000;
learning_rate = 0.005;
%Transformation Matrix
mo = 1/16;
T = zeros(N,784);
t1 = 0;
t2 = 0;
for i = 0:48
    if mod(i,7) == 0 %correct μολις φτασει 8η γραμμη(τρεχω απο το 0 αρα για 
                     % i = 7 ειμαι στην 8η επαναληψη αρα 8η γραμμη του Τ
        t1 = 0;
        t2 = 4 * int16(i/7) * 28; %correct
    end
    
    for j = 0*28 : 28 : 3*28
        T(  i+1, (j+1) + t2 + t1  : (j+4) + t2 + t1 ) = mo;
    end
    t1 = t1+4;
end


% Z1 = load("Z1.mat");
% Z2 = load("Z2.mat");
% Z3 = load("Z3.mat");
% Z4 = load("Z4.mat");

%Zintilial = Z2.Zall(:,3); %Input from file Z.Zall(:,i) /{i = 1,2,3,4,5,6,7}/ 
                          % where i choose the best initial Z for the algorithm
Zintilial = randn(10,1);

lamb = 1;    %ADAM constant
P = 0;       %ADAM Power
c = 10^(-6); %ADAM small number
Z = Zintilial;%Random Input

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
imshow(kron(reshape(Xn(:,eight),7,7),ones(4,4))); 
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