function [T_recovered, S_recovered, C, S_omega, X_omega ] = nmf_tps(T, T_db, O, r, use_dB, Sc, Ctrue)
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
    
    %% Reconstruct spatial loss field for each emitter from the S matrix
    S_tensor = zeros(r, I*J);
    j = 1;
    for i=1:I*J
        if O(i)
            S_tensor(:,i) = S(:,j);
            j = j+1;
        end
    end
    S_tensor = mat2tens(S_tensor,[I J r], 3);

    O_mat = reshape(O,[I,J]);

    Strue = zeros(I,J,r);
    for rr=1:r
        Strue(:,:,rr) = Sc{rr};
    end

%     strc = '';
%     if structured_c
%         strc = '_structured';
%     end
%     freq_bands = '';
%     if K ~= 64
%         freq_bands = strcat('_', string(K))
%     end
%     save(strcat('data/full_tensor/s_sampled/',string(file_count),'_',string(r),'_',string(shadowSigma),'_',string(Xc),'_',string(f),'_',string(snr), strc, freq_bands,'.mat'), 'S_tensor');
%     save(strcat('data/full_tensor/s_true/',string(file_count),'_',string(r),'_',string(shadowSigma),'_',string(Xc),'_',string(f),'_',string(snr), strc, freq_bands,'.mat'), 'Strue'); 
%     save(strcat('data/full_tensor/x_true/',string(file_count),'_',string(r),'_',string(shadowSigma),'_',string(Xc),'_',string(f),'_',string(snr), strc, freq_bands,'.mat'), 'T');
%     save(strcat('data/full_tensor/c_recovered/',string(file_count),'_',string(r),'_',string(shadowSigma),'_',string(Xc),'_',string(f),'_',string(snr), strc, freq_bands,'.mat'), 'C');
%     save(strcat('data/full_tensor/c_true/',string(file_count),'_',string(r),'_',string(shadowSigma),'_',string(Xc),'_',string(f),'_',string(snr), strc, freq_bands,'.mat'), 'Ctrue');
%     save(strcat('data/full_tensor/omega/',string(file_count),'_',string(r),'_',string(shadowSigma),'_',string(Xc),'_',string(f),'_',string(snr), strc, freq_bands,'.mat'), 'O_mat');

    % % TPS  % %
    
    S_tps = zeros(I,J,r);
    
    for rr = 1:r
        X_coord = [];
        Y_coord = [];
        Z = [];
        O_mat = reshape(O, [I,J]);

        for j=1:51
            for i=1:51
                if O_mat(i,j)
                    X_coord = [X_coord; i];
                    Y_coord = [Y_coord; j];
                    Z = [Z; S_tensor(i,j,rr)];
                end
            end
        end
        x_grid = 1:I;
        y_grid = 1:J;
        [x,y] = meshgrid(y_grid, x_grid);
        lambda = 1e-2;
        z = TPS(X_coord, Y_coord,Z, y, x, lambda);
        S_tps(:,:,rr) = reshape(z, [I J]);
    end
    
    S_recovered = S_tps;
    
    % fprintf('NMF_TPS elapsed time is: %.2f seconds. \n',toc')

    Ctrue_n = ColumnNormalization(Ctrue);
    naec = 0;
    for rr = 1:r
        norm_1c = Ctrue_n(:,rr)/sum(abs(Ctrue_n(:,rr))) - C(:,rr)/sum(abs(C(:,rr)));
        naec = naec + sum(abs(norm_1c));
    end
    % NAEC_tps = naec/r

    for rr = 1:r
        minSc{rr} = min(Sc{rr},[],'all');
        S_recovered(:,:,rr) = S_recovered(:,:,rr)/norm(S_recovered(:,:,rr),'fro');
        S_temp = S_recovered(:,:,rr);
        S_temp(S_recovered(:,:,rr)<minSc{rr}) = minSc{rr};
        S_recovered(:,:,rr) = S_temp;
    end
    
    naes = 0;
    for rr = 1:1
        norm_1s = Sc{rr}/sum(abs(Sc{rr}),'all') - S_recovered(:,:,rr)/sum(abs(S_recovered(:,:,rr)),'all');
        naes = naes + sum(abs(norm_1s),'all');
    end
    % NAES_tps = naes/1

    %% reconstruct the recovered tensor
    T_recovered = zeros([I J K]);
    for i=1:r
        T_recovered = T_recovered + outprod(S_recovered(:,:,i), C(:,i));
    end

    % NAEX_tps = sum(abs(T-T_recovered),'all')./sum(abs(T),'all')
    
%     save(strcat('data/full_tensor/x_tps/',string(file_count),'_',string(r),'_',string(shadowSigma),'_',string(Xc),'_',string(f),'_',string(snr), strc, freq_bands,'.mat'), 'T_recovered');


end