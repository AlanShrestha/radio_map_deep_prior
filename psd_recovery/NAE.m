function error = NAE(T, T_recovered)
    error = sum(abs(T/sum(abs(T),'all') - T_recovered/sum(abs(T_recovered),'all')), 'all');
end
