Initial estimate S and C through NMF and deep prior
Joint Optimization:
    Use the S(k) estimate to improve C(k+1) throgh min norm(X - C(k)S(k))
    Use C(k+1) to estimate S(k+1) 