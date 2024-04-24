pragma circom 2.0.0;

include "/Users/jernkun/Desktop/zk-stats-lib/zkstats/onnx2circom/mpc.circom";

template Model() {
signal input in[1][1];
signal input tf_log_1_out[1];
signal output out[1];

component tf_log_1 = TFLog(2);

for (var i0 = 0; i0 < 1; i0++) {
    for (var i1 = 0; i1 < 1; i1++) {
        tf_log_1.in[i0][i1] <== in[i0][i1];
}}
for (var i0 = 0; i0 < 1; i0++) {
    tf_log_1.out[i0] <== tf_log_1_out[i0];
}
for (var i0 = 0; i0 < 1; i0++) {
    out[i0] <== tf_log_1.out[i0];
}

}

component main = Model();
