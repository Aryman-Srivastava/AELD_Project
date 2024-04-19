#include "helper.h"

void multiplyMatrixVector_48_104(float matrix[VAR_48][VAR_104], float vector[VAR_104], float result[VAR_48]) {
    for (int i = 0; i < VAR_48; i++) {
        result[i] = 0;
        #pragma HLS PIPELINE
        for (int j = 0; j < VAR_104; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void multiplyMatrixVector_48_48(float matrix[VAR_48][VAR_48], float vector[VAR_48], float result[VAR_48]) {
    for (int i = 0; i < VAR_48; i++) {
        result[i] = 0;
#pragma HLS PIPELINE
        for (int j = 0; j < VAR_48; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}
void multiplyMatrixVector_96_48(float matrix[VAR_96][VAR_48], float vector[VAR_48], float result[VAR_48]) {
    for (int i = 0; i < VAR_96; i++) {
        result[i] = 0;
#pragma HLS PIPELINE
        for (int j = 0; j < VAR_48; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void resetGate(float Wir[VAR_48][VAR_104], float X[VAR_104], float bir[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float r[VAR_48]) {
    float r1[VAR_48], r2[VAR_48];
    multiplyMatrixVector_48_104(Wir, X, r1);
    multiplyMatrixVector_48_48(Wt, ht, r2);
    reset_gate_loop: for(int i = 0; i < VAR_48; i++){
#pragma HLS PIPELINE
        r[i] = sigmoid(r1[i] + r2[i] + bir[i] + bt[i]);

    }
}

void updateGate(float Wiz[VAR_48][VAR_104], float X[VAR_104], float biz[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float z[VAR_48]) {
    float z1[VAR_48], z2[VAR_48];
    multiplyMatrixVector_48_104(Wiz, X, z1);
    multiplyMatrixVector_48_48(Wt, ht, z2);
    update_gate_loop: for(int i = 0; i < VAR_48; i++){
#pragma HLS PIPELINE
        z[i] = sigmoid(z1[i] + z2[i] + biz[i] + bt[i]);

    }
}


void tanh_layer(float Win[VAR_48][VAR_104], float X[VAR_104], float bin[VAR_48], float r[VAR_48], float Whn[VAR_48][VAR_48], float ht_1[VAR_48], float bh[VAR_48], float n[VAR_48]) {
    float n1[VAR_48], n2[VAR_48],n_int[VAR_48];
    multiplyMatrixVector_48_104(Win, X, n1);
    multiplyMatrixVector_48_48(Whn, ht_1, n2);
    tanh_gate_loop: for(int i = 0; i < VAR_48; i++){
#pragma HLS PIPELINE
        n_int[i] = (n2[i] + bh[i]) * r[i] + (n1[i] + bin[i]);
        n[i] = tanh(n_int[i]);
    }
}

void ht_new(float z[VAR_48], float ht_1[VAR_48], float n[VAR_48], float ht[VAR_48]) {
    ht_new_loop: for(int i = 0; i < VAR_48; i++){
#pragma HLS PIPELINE
        ht[i] = (1 - z[i]) * n[i] + z[i] * ht_1[i];
    }
}
