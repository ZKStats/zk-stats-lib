pragma circom 2.0.0;


template TFAdd() {
    signal input in[2][1];
    signal output out[1];

    out[0] <== in[0][0] + in[1][0];
}

template TFSub() {
    signal input in[2][1];
    signal output out[1];

    out[0] <== in[0][0] - in[1][0];
}

template TFMul() {
    signal input in[2][1];
    signal output out[1];

    out[0] <== in[0][0] * in[1][0];
}

template TFDiv() {
    signal input in[2][1];
    signal output out[1];

    out[0] <== in[0][0] / in[1][0];
}

template TFReduceSum(nInputs) {
    signal input in[nInputs][1];
    signal output out[1];

    signal sum_till[nInputs];
    sum_till[0] <== in[0][0];
    component add[nInputs-1];
    for (var i = 0; i<nInputs-1; i++){
        add[i] = TFAdd();
        add[i].in[0][0] <== sum_till[i];
        add[i].in[1][0] <== in[i+1][0];
        sum_till[i+1] <== add[i].out[0];
    }
    // FIXME: adding 0 is a workaround for nInputs=1, to force `in[0][0]` to be an
    // input in a gate
    out[0] <== sum_till[nInputs-1] + 0;
}

template TFReduceMean(nInputs) {
    signal input in[nInputs][1];
    signal output out[1];

    component sum = TFReduceSum(nInputs);
    for (var i = 0; i<nInputs; i++){
        sum.in[i][0] <== in[i][0];
    }

    component div = TFDiv();
    div.in[0][0] <== sum.out[0];
    div.in[1][0] <== nInputs;
    out[0] <== div.out[0];
}


// TODO: e should be 2.71828 instead of 2 for now
template TFLog(e) {
    signal input in[1][1];

    signal output out[1];

    // Approximate natural log with talyer series. For 0 < x <= 2
    // - ln(x) = ln(1 + (x-1)) = 0 + (x-1) - (x-1)^2/2 + (x-1)^3/3 - (x-1)^4/4 + ...
    // - To ensure x <= 2, we can use the following property of logarithm:
    // ln(x) = ln(x / b) + ln(b) where b is an integer e^k. So, we need to calculate
    //  - Step 1: b = e^k such that x/b <= 1
    //  - Step 2: ln(x/b) using talyer series
    //  - Step 3: ln(x) = ln(x/b) + ln(b) = ln(x/b) + k

    // x can be only up to e^max_exponent
    var max_exponent = 64;
    var taylor_series_iterations = 40;

    // Step 1: Find b so that b = e^k and x/b <= 1
    // find b so that x/b < 1 and can be used in talyer series
    // b = e^k, e=2.71828
    signal x <== in[0][0];
    // e_until[i] = e^i
    signal e_until[max_exponent];
    e_until[0] <== 1;
    for (var i = 1; i < max_exponent; i++) {
        e_until[i] <== e_until[i-1] * e;
    }

    // e^k >= x and e^(k-1) < x
    // sel : [0, 0, 1, 0, 0]
    // k   : [0, 1, 2, 3, 4]
    // sum(sel*k) = k
    signal sel[max_exponent];
    sel[0] <== x <= 1;
    component sel_comp[max_exponent];
    for (var i = 1; i < max_exponent; i++) {
        sel_comp[i] = TFMul();
        sel_comp[i].in[0][0] <== x > e_until[i-1];
        sel_comp[i].in[1][0] <== x <= e_until[i];
        sel[i] <== sel_comp[i].out[0];
    }
    component k_by_sum = TFReduceSum(max_exponent);
    for (var i = 0; i < max_exponent; i++) {
        k_by_sum.in[i][0] <== sel[i] * i;
    }
    signal k <== k_by_sum.out[0];
    // sum(sel*e^k) = b
    component b_by_sum = TFReduceSum(max_exponent);
    for (var i = 0; i < max_exponent; i++) {
        b_by_sum.in[i][0] <== sel[i] * e_until[i];
    }
    signal b <== b_by_sum.out[0];

    // Step 2: Calculate ln(x/b) using talyer series
    signal x_over_b <== x / b;

    signal x_over_b_minus_one_exp[taylor_series_iterations];
    x_over_b_minus_one_exp[0] <== 0;
    x_over_b_minus_one_exp[1] <== (x / b) - 1;
    for (var i = 2; i < taylor_series_iterations; i++) {
        x_over_b_minus_one_exp[i] <== x_over_b_minus_one_exp[i-1] * (1 - x_over_b);
    }

    signal taylor_series[taylor_series_iterations];
    taylor_series[0] <== 0;
    for (var i = 1; i < taylor_series_iterations; i++) {
        taylor_series[i] <== x_over_b_minus_one_exp[i] / i;
    }

    signal taylor_series_sum;
    component taylor_series_sum_comp = TFReduceSum(taylor_series_iterations);
    for (var i = 0; i < taylor_series_iterations; i++) {
        taylor_series_sum_comp.in[i][0] <== taylor_series[i];
    }
    taylor_series_sum <== taylor_series_sum_comp.out[0];

    out[0] <== taylor_series_sum + k;
}
