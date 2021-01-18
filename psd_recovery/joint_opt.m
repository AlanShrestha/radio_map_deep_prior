% Implementation of Joint optimization 
% This code uses capital letter such as 'A' with small m appended to denote the matricized version 'Am' 
% And Av to denote the vectorized version of the matrix A of Am

clear;
%% Generate a sample of radio map

R = 2;    % number of emitters
f = 0.1;  % fraction of sampled data
K=64;     % length of spectrum
Xc = 50;  % Correlation Distance
shadow_sigma = 5;       % shadowing variance
snr = 0;                % SNR
structure_c = false;    % whether to generate the radio map using pre-determined emitter locations
z_dimension = 256;

gd_lr = 0.01;

% generate radio map
% the last argument determines the type of psd basis function 's': sinc 'g': gaussian
[T, Sc, C_true] = generate_data_for_comparison(false, K, R, shadow_sigma, Xc, structure_c, 'g'); 


%% Initialization

% load python modules
py.sys.setdlopenflags(int32(10));

% Use NMF factorization to get C and S from the sampled tensor

[I,J,K] = size(T);
IJ = I*J;
num_samples = round(f*IJ);
Omega = randperm(IJ, num_samples)';

% sampling matrix
Ov = false(1,IJ);
Ov(Omega) = true;

% Mode-3 matrix unfolding, arrange fibers as columns of the matrix from tensor
Tm = tens2mat(T,3);
Tm_omega = Tm(:, Ov);

% apply SPA algorithm to obtain the indices of factor C
indices_C = SPA(Tm_omega, R);
C = Tm_omega(:, indices_C);

%% remove permutation
[cpderrc,per,~]=cpderr(C_true,C);
C_noperm = C*per;
C_p = ColumnPositive(C_noperm);
C_p(C_p<0)=0;
C = ColumnNormalization(C_p);

% obtain S matrix whose rows are the spatial loss field for emitters
pseudo_inverse_C = pseudo_inverse(C);
Sm_omega = pseudo_inverse_C*Tm_omega;

S_true = zeros(I,J,R);
for rr=1:R
    S_true(:,:,rr) = Sc{rr};
end
C_true = ColumnNormalization(C_true);

%% Reconstruct spatial loss field for each emitter from the Sm_omega matrix
S_omega = zeros(R, I*J);
j = 1;
for i=1:I*J
    if Ov(i)
        S_omega(:,i) = Sm_omega(:,j);
        j = j+1;
    end
end
S_omega = mat2tens(S_omega,[I J R], 3);

Om = reshape(Ov,[I,J]);

W_py = py.numpy.array(Om);
S_py = py.numpy.array(S_omega);
S_ae = py.nn_descent_ae.model(S_py, W_py, R);
S_ae = double(S_ae);

steps = 1;

%% reconstruct the recovered tensor
T_ae = get_tensor(S_ae, C);
Cost_before_opt = Cost(T, T_ae, Om)
NAE_before_opt = NAE(T,T_ae)

cost_list = [Cost_before_opt];
nae_list = [NAE_before_opt];

z = zeros(R, z_dimension);
%% Start joint Optimization of C and S 
for step=1:steps

    %% Step 1
    % Cr optimization subprobplem.
    
    % find the optimal learning rate param alpha
    % e = eig(Sm_omega*Sm_omega');
    % lambda_max = max(e);
    % alpha = 1/lambda_max;
    
    % % descent directioin of C 
    % del_fc = (C*Sm_omega*Sm_omega' - Tm_omega*Sm_omega');
    
    % run decent on C
    % for i=1:R 
    %     C = C - alpha*del_fc;
    % end
    
    
    %% Step 2
    % Stheta optimization subproblem

    W_py = py.numpy.array(Om);
    X_py = py.numpy.array(T);
    C_py = py.numpy.array(C);
    S_py = py.numpy.array(S_omega);
    S_true_py = py.numpy.array(S_true);
    
    % save files
    save('data/T.mat', 'T');
    save('data/C.mat', 'C');
    save('data/Om.mat', 'Om');
    save('data/S_omega.mat', 'S_omega');
    save('data/Sc.mat', 'S_true');

    % Call the nn gradient descent optimizer: returns optimized S_omega
    % if first step then find the initial z vector
    step
    if step == 1
        tuple = py.nn_descent_gan.inverse_gan2(W_py, X_py, S_py, C_py, R, S_true_py, gd_lr);
        cell_tuple = cell(tuple);
        z_py = cell_tuple{1};
        S_comp = cell_tuple{2};
        z = double(z_py);
    else
        z_py = py.numpy.array(z);
        tuple = py.nn_descent_gan.optimize_z(W_py, X_py, S_py, C_py, R, z_py, gd_lr);
        cell_tuple = cell(tuple);
        z_py = cell_tuple{1};
        S_comp = cell_tuple{2};
        z = double(z_py);
    end

    S_gan = double(S_comp);
    S_omega = S_gan.*Om;
    temp = tens2mat(S_gan,3);
    Sm_omega = temp(:, Ov);
    
    % for rr=1:R
    %     S_gan(:,:,rr) = S_gan(:,:,rr)/norm(S_gan(:,:,rr),'fro');
    % end
    
    %% Cost
    T_recovered = get_tensor(S_gan, C);
    t_recovered_sum = sum(T_recovered, 'all')
    w_sum = sum(Om, 'all')
    T_sum = sum(T, 'all')
    Cost_after_nn_descent = Cost(T, T_recovered, Om)
    cost_list = [cost_list Cost_after_nn_descent];
    
    
    NAES_gan = NAE(S_true, S_gan)
    NAES_ae = NAE(S_true, S_ae)
    
    NAE_after_opt = NAE(T,T_recovered)
    nae_list = [nae_list NAE_after_opt];
    
end
S_recovered_after_opt = S_gan;

%% plot
figure(1);
subplot(211);
plot(1:length(cost_list), cost_list);
xlabel("iterations");
ylabel("cost");
title("Objective function (10% sampling, R=5, sigma=10, Xc=30), without constraint");
subplot(212);
plot(1:length(nae_list), nae_list);
title("MSE");
xlabel("iterations");
ylabel("MSE");

r = 1;
figure(2);
subplot(131);
contourf(10*log10(S_true(:,:,r)), 100, 'linecolor', 'None');
% title("true slf, 10% sampling, R=5, sigma=10, Xc=30");
title('true slf');
colormap jet;
subplot(132);
contourf(10*log10(S_ae(:,:,r)), 100, 'linecolor', 'None');
title("slf autoencoder completion");
colormap jet;
subplot(133);
contourf(10*log10(S_recovered_after_opt(:,:,r)), 100, 'linecolor', 'None');
title("GAN optimization");
colormap jet;


%% functions
function error = total_cost(T, T_recovered)
    error = frob(T - T_recovered)^2;
end

function error = NAE(T, T_recovered)
    error = sum(abs(T/sum(abs(T),'all') - T_recovered/sum(abs(T_recovered),'all')), 'all');
end

function error = Cost(T, T_recovered, Om)
    error = sum((Om.*T - Om.*T_recovered).^2, 'all');
end

function X = get_tensor(S_omega, C)
    sizec = size(C);
    X = zeros(51,51,sizec(1));

    for rr=1:sizec(2)
        X = X + outprod(S_omega(:,:,rr), C(:,rr));
   end
end