% Implementation of Joint optimization 

%% Initialization
% load python modules
py.sys.setdlopenflags(int32(10));

% Use NMF factorization to get C and S from the sampled tensor
R = 5;
r = R;
% fraction of sampled data
f = 0.1;
K=64;
Xc = 30;
shadow_sigma = 10;
snr = 0;
structure_c = false;

% lowner rank 
L = 8;

[T, Sc, Ctrue] = generate_data_for_comparison(false, K, R, shadow_sigma, Xc, structure_c, 'g');
% [T_db, Strue_db, Ctrue_db] = generate_data(true);
T_db = 10*log10(T + 1e-6);

% flags to plot data
plot_slf = false;
plot_radio_map = false;
plot_psd = true;
plot_db = false;
plot_tps = true;
use_dB = false;

[I,J,K] = size(T);
IJ = I*J;
num_samples = round(f*IJ);
Omega = randperm(IJ, num_samples)';

O = false(1,IJ);
O(Omega) = true;

% [T_tps, S_tps, C_tps, S_omega, X_omega] = nmf_tps(T, T_db, O, R, use_dB, Sc, Ctrue);

tic
index = @(matrix, r, c) matrix(r,c);

[I, J, K] = size(T);

% Mode-3 matrix unfolding, arrange fibers as columns of the matrix from
% tensor
T3 = tens2mat(T,3);
T3_db = tens2mat(T_db,3);

X_raw = T3(:, O);
X_omega = X_raw;
X_raw_db = T3_db(:,O);
X = X_raw;
X_db = X_raw_db;

% apply SPA algorithm to obtain the indices of factor C
indices_C = SPA(X, r);
C = X(:, indices_C);

%% remove permutation
[cpderrc,per,~]=cpderr(Ctrue,C);
C_noperm = C*per;

C_p = ColumnPositive(C_noperm);
C_p(C_p<0)=0;
C = ColumnNormalization(C_p);


% obtain S matrix whose rows are the spatial loss field for emitters
pseudo_inverse_C = pseudo_inverse(C);

if use_dB
    S = pseudo_inverse_C*X_db;
else
    S = pseudo_inverse_C*X;
end
S_omega = S;

% reconstruct X    
X_recovered = C*S;

Strue = zeros(I,J,r);
for rr=1:r
    Strue(:,:,rr) = Sc{rr};
end
Ctrue_n = ColumnNormalization(Ctrue);

%% Reconstruct spatial loss field for each emitter from the S matrix
S_tensor = zeros(r, I*J);
j = 1;
for i=1:I*J
    if O(i)
        S_tensor(:,i) = S_omega(:,j);
        j = j+1;
    end
end
S_tensor = mat2tens(S_tensor,[I J r], 3);

O_mat = reshape(O,[I,J]);

W_py = py.numpy.array(O_mat);
S_py = py.numpy.array(S_tensor);
S_comp = py.nn_descent.model(S_py, W_py, R);
S_recovered = double(S_comp);
S_recovered_before_opt = S_recovered;
%% reconstruct the recovered tensor
T_recovered = zeros([I J K]);
for i=1:r
    T_recovered = T_recovered + outprod(S_recovered(:,:,i), C(:,i));
end

% NAEX_tps = sum(abs(T-T_recovered),'all')./sum(abs(T),'all')

NAE_before_opt = NAE(T,T_recovered)


S_result = S_tensor;
S_tensor = S_recovered;
steps = 200;

% T_recovered = get_tensor(S_tensor, C);
Cost_before_opt = Cost(T, T_recovered, O_mat)

cost_list = [Cost_before_opt];
nae_list = [NAE_before_opt];

for step=1:steps
    %% Step 1
    % Cr optimization subprobplem.
    
    e = eig(S_omega*S_omega');
    lambda_max = max(e);
    alpha = 1/lambda_max;
    
    % T_recovered = get_tensor(S_tensor, C);
    % Cost_before_c_opt = Cost(T, T_recovered, O_mat);
    
    % descent directioin 
    del_fc = (C*S_omega*S_omega' - X_omega*S_omega');
    for i=1:R 
        C = C - alpha*del_fc;
    end
    
    T_recovered = get_tensor(S_tensor, C);
%     Cost_after_c_opt = Cost(T, T_recovered, O_mat);
%     cost_list = [cost_list Cost_after_c_opt];
    NAE_after_c_opt = NAE(T, T_recovered)
    % nae_list = [nae_list NAE_after_c_opt];

    % naec = 0;
    % for rr = 1:R
    %     norm_1c = Ctrue(:,rr)/sum(abs(Ctrue(:,rr))) - C(:,rr)/sum(abs(C(:,rr)));
    %     naec = naec + sum(abs(norm_1c));
    % end
    % NAEC_total = naec/R
%     
    truncC = [];
    for rr=1:R
        lownerC = loewnerize(C(:,rr));
        [U,S,V] = svd(lownerC);
        diagS = diag(S);
        diagS(L :end) = 0;
        truncS = diag(diagS);
        trunc_lownerC = U*truncS*V';
        truncCr = deloewnerize(trunc_lownerC);
        truncC = [truncC truncCr(:,1)];
    end

%     C = truncC;
    naec = 0;
    
    for rr = 1:R
        truncC(:,rr) = truncC(:,rr) - min(truncC(:,rr));
    end
    truncC = ColumnNormalization(truncC);
    
    for rr = 1:R
        norm_1c = Ctrue(:,rr)/sum(abs(Ctrue(:,rr))) - truncC(:,rr)/sum(abs(truncC(:,rr)));
        naec = naec + sum(abs(norm_1c));
    end
    NAEC_truncated = naec/R
    figure(1);
    subplot(311);
    plot(1:64, Ctrue_n(:,1));
    subplot(312);
    plot(1:64, C(:,1));
    subplot(313);
    plot(1:64, truncC(:,1), 1:64, Ctrue_n(:,1));
    plot(1:64, Ctrue_n(:,1));
    
    C = truncC;
%     
%     T_recovered = get_tensor(S_tensor, C);
    % Cost_after_truncation = Cost(T, T_recovered, O_mat);
    % cost_list = [cost_list Cost_after_truncation];


    %% Step 2
    % Stheta optimization subproblem
    W_py = py.numpy.array(O_mat);
    X_py = py.numpy.array(T);
    C_py = py.numpy.array(C);

    S_py = py.numpy.array(S_result);

    % save files
    save('data/T.mat', 'T');
    save('data/C.mat', 'C');
    save('data/O_mat.mat', 'O_mat');
    save('data/S_tensor.mat', 'S_tensor');
    
    % call the nn gradient descent optimizer: returns optimized S_tensor
    S_descent = py.nn_descent.run_descent(W_py, X_py, S_py, C_py, R);
    S_result = double(S_descent);

    S_comp = py.nn_descent.model(S_descent, W_py, R);
    S_tensor = double(S_comp);
    
    S_omega_full = tens2mat(S_tensor, 3);
    S_omega = S_omega_full(:,O);
    
    
    %% Cost
    T_recovered = get_tensor(S_tensor, C);
    Cost_after_nn_descent = Cost(T, T_recovered, O_mat);
    cost_list = [cost_list Cost_after_nn_descent];
    
    NAE_after_opt = NAE(T,T_recovered)
    nae_list = [nae_list NAE_after_opt];
    
end
S_recovered_after_opt = S_tensor;

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

r = 2;
figure(2);
subplot(131);
contourf(10*log10(Strue(:,:,r)), 100, 'linecolor', 'None');
title("true slf, 10% sampling, R=5, sigma=10, Xc=30");
colormap jet;
subplot(132);
contourf(10*log10(S_recovered_before_opt(:,:,r)), 100, 'linecolor', 'None');
title("slf before optimization");
colormap jet;
subplot(133);
contourf(10*log10(S_recovered_after_opt(:,:,r)), 100, 'linecolor', 'None');
title("slf after optimization");
colormap jet;


%% functions
function error = NAE(T, T_recovered)
    error = frob(T - T_recovered)^2;
end

function error = Cost(T, T_recovered, O_mat)
    error = frob(O_mat.*T - O_mat.*T_recovered)^2;
end

function X = get_tensor(S_tensor, C)
    sizec = size(C);
    X = zeros(51,51,sizec(1));

    for rr=1:sizec(2)
        X = X + outprod(S_tensor(:,:,rr), C(:,rr));
   end
end