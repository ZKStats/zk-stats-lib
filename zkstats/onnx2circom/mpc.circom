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
    out[0] <== sum_till[nInputs-1];
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

template TFLog(e) {
    signal input in[1][1];
    // b must be passed in as secret

    signal output out[1];

    var upper_bound = 64;
    // find b so that b = e^k and x/b <= 1
    // find b so that x/b < 1 and can be used in talyer series
    // b = e^k, e=2.71828
    signal e_until[upper_bound];
    e_until[0] <== 1;
    for (var i = 1; i < upper_bound; i++) {
        e_until[i] <== e_until[i-1] * e;
    }
    signal x <== in[0][0];
    // find k s.t. e_until[k] >= x and e_until[k-1] < x
    // signal k;

    // e^k >= x and e^(k-1) < x
    // sel : [0, 0, 1, 0, 0]
    // k   : [0, 1, 2, 3, 4]
    // sum(sel*k) = k

    signal sel[upper_bound];
    sel[0] <== x <= 1;
    component sel_comp[upper_bound];
    for (var i = 1; i < upper_bound; i++) {
        sel_comp[i] = TFMul();
        sel_comp[i].in[0][0] <== x > e_until[i-1];
        sel_comp[i].in[1][0] <== x <= e_until[i];
        sel[i] <== sel_comp[i].out[0];
    }

    // component sum = TFReduceSum(nInputs);
    // for (var i = 0; i<nInputs; i++){
    //     sum.in[i][0] <== in[i][0];
    // }

    component k_by_sum = TFReduceSum(16);
    // for (var i = 0; i < 4; i++) {
    //     k_by_sum.in[i][0] <== sel[i] * i;
    // }
    k_by_sum.in[0][0] <== sel[0] * 0;
    k_by_sum.in[1][0] <== sel[1] * 1;
    k_by_sum.in[2][0] <== sel[2] * 2;
    k_by_sum.in[3][0] <== sel[3] * 3;
    k_by_sum.in[4][0] <== sel[4] * 4;
    k_by_sum.in[5][0] <== sel[5] * 5;
    k_by_sum.in[6][0] <== sel[6] * 6;
    k_by_sum.in[7][0] <== sel[7] * 7;
    k_by_sum.in[8][0] <== sel[8] * 8;
    k_by_sum.in[9][0] <== sel[9] * 9;
    k_by_sum.in[10][0] <== sel[10] * 10;
    k_by_sum.in[11][0] <== sel[11] * 11;
    k_by_sum.in[12][0] <== sel[12] * 12;
    k_by_sum.in[13][0] <== sel[13] * 13;
    k_by_sum.in[14][0] <== sel[14] * 14;
    k_by_sum.in[15][0] <== sel[15] * 15;

    signal k <== k_by_sum.out[0];

    // out[0] <== k_by_sum.out[0];
    // component k_by_sum = TFReduceSum(upper_bound);
    // component k_by_sum_mul[upper_bound];
    // for (var i = 0; i < upper_bound; i++) {
    //     k_by_sum_mul[i] = TFMul();
    //     k_by_sum_mul[i].in[0][0] <== sel[i];
    //     k_by_sum_mul[i].in[1][0] <== i;
    //     k_by_sum.in[i][0] <== k_by_sum_mul[i].out[0];
    // }
    // out[0] <== k_by_sum.out[0];

    // component b_by_sum = TFReduceSum(upper_bound);
    // for (var i = 0; i < upper_bound; i++) {
    //     b_by_sum.in[i][0] <== sel[i] * e_until[i];
    // }
    // signal b <== b_by_sum.out[0];

    component b_by_sum = TFReduceSum(16);
    b_by_sum.in[0][0] <== sel[0] * e_until[0];
    b_by_sum.in[1][0] <== sel[1] * e_until[1];
    b_by_sum.in[2][0] <== sel[2] * e_until[2];
    b_by_sum.in[3][0] <== sel[3] * e_until[3];
    b_by_sum.in[4][0] <== sel[4] * e_until[4];
    b_by_sum.in[5][0] <== sel[5] * e_until[5];
    b_by_sum.in[6][0] <== sel[6] * e_until[6];
    b_by_sum.in[7][0] <== sel[7] * e_until[7];
    b_by_sum.in[8][0] <== sel[8] * e_until[8];
    b_by_sum.in[9][0] <== sel[9] * e_until[9];
    b_by_sum.in[10][0] <== sel[10] * e_until[10];
    b_by_sum.in[11][0] <== sel[11] * e_until[11];
    b_by_sum.in[12][0] <== sel[12] * e_until[12];
    b_by_sum.in[13][0] <== sel[13] * e_until[13];
    b_by_sum.in[14][0] <== sel[14] * e_until[14];
    b_by_sum.in[15][0] <== sel[15] * e_until[15];

    signal b <== b_by_sum.out[0];

    signal x_over_b <== x / b;

    var taylor_series_iterations = 40;
    signal x_over_b_minus_one_exp[taylor_series_iterations+1];
    x_over_b_minus_one_exp[0] <== 0;
    x_over_b_minus_one_exp[1] <== (x / b) - 1;
    for (var i = 2; i < taylor_series_iterations+1; i++) {
        x_over_b_minus_one_exp[i] <== x_over_b_minus_one_exp[i-1] * (1 - x_over_b);
    }
    // out[0] <== x_over_b_minus_one_exp[taylor_series_iterations];

    out[0] <== k + x_over_b_minus_one_exp[1]+x_over_b_minus_one_exp[2]/2+x_over_b_minus_one_exp[3]/3+x_over_b_minus_one_exp[4]/4+x_over_b_minus_one_exp[5]/5+x_over_b_minus_one_exp[6]/6 + x_over_b_minus_one_exp[7]/7 + x_over_b_minus_one_exp[8]/8 + x_over_b_minus_one_exp[9]/9 + x_over_b_minus_one_exp[10]/10+x_over_b_minus_one_exp[11]/11+x_over_b_minus_one_exp[12]/12+x_over_b_minus_one_exp[13]/13+x_over_b_minus_one_exp[14]/14+x_over_b_minus_one_exp[15]/15+x_over_b_minus_one_exp[16]/16 + x_over_b_minus_one_exp[17]/17 + x_over_b_minus_one_exp[18]/18 + x_over_b_minus_one_exp[19]/19 + x_over_b_minus_one_exp[20]/20+x_over_b_minus_one_exp[21]/21+x_over_b_minus_one_exp[22]/22+x_over_b_minus_one_exp[23]/23+x_over_b_minus_one_exp[24]/24+x_over_b_minus_one_exp[25]/25+x_over_b_minus_one_exp[26]/26+x_over_b_minus_one_exp[27]/27+x_over_b_minus_one_exp[28]/28+x_over_b_minus_one_exp[29]/29+x_over_b_minus_one_exp[30]/30+x_over_b_minus_one_exp[31]/31+x_over_b_minus_one_exp[32]/32+x_over_b_minus_one_exp[33]/33+x_over_b_minus_one_exp[34]/34+x_over_b_minus_one_exp[35]/35+x_over_b_minus_one_exp[36]/36+x_over_b_minus_one_exp[37]/37+x_over_b_minus_one_exp[38]/38+x_over_b_minus_one_exp[39]/39+x_over_b_minus_one_exp[40]/40;
    // out[0] <== k;

    // signal taylor_series[taylor_series_iterations+1];
    // taylor_series[0] <== 0;
    // for (var i = 1; i < taylor_series_iterations+1; i++) {
    //     taylor_series[i] <== x_over_b_minus_one_exp[i] / i;
    // }
    // out[0] <== k + taylor_series[0]+taylor_series[1]+taylor_series[2]+taylor_series[3]+taylor_series[4]+taylor_series[5]+taylor_series[6]+taylor_series[7]+taylor_series[8]+taylor_series[9]+taylor_series[10]+taylor_series[11]+taylor_series[12]+taylor_series[13]+taylor_series[14]+taylor_series[15]+taylor_series[16];
    // signal taylor_series_sum;
    // component taylor_series_sum_comp = TFReduceSum(taylor_series_iterations);
    // for (var i = 0; i < taylor_series_iterations; i++) {
    //     taylor_series_sum_comp.in[i][0] <== taylor_series[i+1];
    // }
    // taylor_series_sum <== taylor_series_sum_comp.out[0];
    // out[0] <== taylor_series_sum + k;

    // log(x) = log(x/b) + log(b)
    // use talyer series to approximate
    // log(x) = log(1 + (x-1)) = (x-1) - (x-1)^2/2 + (x-1)^3/3 - (x-1)^4/4 + ...

}
