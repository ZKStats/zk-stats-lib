pragma circom 2.0.0;


template TFAdd(nElements) {
    signal input left[nElements];
    signal input right[nElements];
    signal output out[nElements];
    for (var i = 0; i< nElements; i ++){
        out[i] <== left[i] + right[i];
    }
}

template TFSub(nElements) {
    signal input left[nElements];
    signal input right[nElements];
    signal output out[nElements];
    for (var i = 0; i< nElements; i++){
        out[i] <== left[i] - right[i];
    }
}

template TFMul(nElements) {
    signal input left[nElements];
    signal input right[nElements];
    signal output out[nElements];
    for (var i = 0; i<nElements; i++){
        out[i] <== left[i] * right[i];
    }
}

template TFDiv(nElements) {
    signal input left[nElements];
    signal input right[nElements];
    signal output out[nElements];
    for (var i = 0; i<nElements; i++){
        out[i] <== left[i] / right[i];
    }
}

template TFEqual(nElements) {
    signal input left[nElements];
    signal input right[nElements];
    signal output out[nElements];
    for (var i = 0; i<nElements; i++){
        out[i] <== left[i] == right[i];
    }
}

template TFReduceSum(nInputs) {
    signal input in[nInputs];
    signal output out;

    signal sum_till[nInputs];
    sum_till[0] <== in[0];
    component add[nInputs-1];
    for (var i = 0; i<nInputs-1; i++){
        add[i] = TFAdd(1);
        add[i].left[0] <== sum_till[i];
        add[i].right[0] <== in[i+1];
        sum_till[i+1] <== add[i].out[0];
    }
    // FIXME: adding 0 is a workaround for nInputs=1, to force `in[0][0]` to be an
    // input in a gate
    out <== sum_till[nInputs-1] + 0;
}

template TFReduceMean(nInputs) {
    signal input in[nInputs];
    signal output out;

    component sum = TFReduceSum(nInputs);
    for (var i = 0; i<nInputs; i++){
        sum.in[i] <== in[i];
    }

    out <== sum.out / nInputs;
}


// TODO: e should be 2.71828 instead of 2 for now
template TFLog(e, nInputs) {
    signal input in[nInputs];
    signal output out[nInputs];

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
    signal x[nInputs];
    signal e_until[nInputs][max_exponent];
    signal sel[nInputs][max_exponent];
    // component sel_comp[nInputs][max_exponent];
    component k_by_sum[nInputs];
    signal k[nInputs];
    component b_by_sum[nInputs];
    signal b[nInputs];
    signal x_over_b[nInputs];
    signal x_over_b_minus_one_exp[nInputs][taylor_series_iterations];
    signal taylor_series[nInputs][taylor_series_iterations];
    signal taylor_series_sum[nInputs];
    component taylor_series_sum_comp[nInputs];

    for (var input_index = 0; input_index < nInputs; input_index++){
        x[input_index] <== in[input_index];
        // e_until[i] = e^i
        e_until[input_index][0] <== 1;
        for (var i = 1; i < max_exponent; i++) {
            e_until[input_index][i] <== e_until[input_index][i-1] * e;
        }

        // e^k >= x and e^(k-1) < x
        // sel : [0, 0, 1, 0, 0]
        // k   : [0, 1, 2, 3, 4]
        // sum(sel*k) = k

        sel[input_index][0] <== x[input_index] <= 1;

        for (var i = 1; i < max_exponent; i++) {
            sel[input_index][i] <== (x[input_index] > e_until[input_index][i-1]) * (x[input_index] <= e_until[input_index][i]);
        }
        k_by_sum[input_index] = TFReduceSum(max_exponent);
        for (var i = 0; i < max_exponent; i++) {
            k_by_sum[input_index].in[i] <== sel[input_index][i] * i;
        }
        k[input_index] <== k_by_sum[input_index].out;
        // sum(sel*e^k) = b
        b_by_sum[input_index] = TFReduceSum(max_exponent);
        for (var i = 0; i < max_exponent; i++) {
            b_by_sum[input_index].in[i] <== sel[input_index][i] * e_until[input_index][i];
        }
        b[input_index] <== b_by_sum[input_index].out;

        // Step 2: Calculate ln(x/b) using talyer series
        x_over_b[input_index] <== x[input_index] / b[input_index];


        x_over_b_minus_one_exp[input_index][0] <== 0;
        x_over_b_minus_one_exp[input_index][1] <== (x[input_index] / b[input_index]) - 1;
        for (var i = 2; i < taylor_series_iterations; i++) {
            x_over_b_minus_one_exp[input_index][i] <== x_over_b_minus_one_exp[input_index][i-1] * (1 - x_over_b[input_index]);
        }

        taylor_series[input_index][0] <== 0;
        for (var i = 1; i < taylor_series_iterations; i++) {
            taylor_series[input_index][i] <== x_over_b_minus_one_exp[input_index][i] / i;
        }

        taylor_series_sum_comp[input_index] = TFReduceSum(taylor_series_iterations);
        for (var i = 0; i < taylor_series_iterations; i++) {
            taylor_series_sum_comp[input_index].in[i] <== taylor_series[input_index][i];
        }
        taylor_series_sum[input_index] <== taylor_series_sum_comp[input_index].out;

        out[input_index] <== taylor_series_sum[input_index] + k[input_index];
    }
}
