#include <stdio.h>
#include <stdlib.h>

#include "header_files/matrixMultiplication.h"
#include "header_files/hiddenGateBias.h"
#include "header_files/hiddenGateWeights.h"
#include "header_files/inputGateBias.h"
#include "header_files/inputGateWeights.h"
#include "header_files/linearLayerBias.h"
#include "header_files/linearLayerWeight.h"
#include "header_files/input_0.h"
#include "header_files/input_5.h"
#include "header_files/input_10.h"
#include "header_files/input_15.h"
#include "header_files/input_20.h"
#include "header_files/input_25.h"
#include "header_files/input_30.h"
#include "header_files/input_35.h"
#include "header_files/input_40.h"


#define VAR1 104
#define VAR2 48
#define VAR3 52
#define VAR4 96
#define FRAME_SIZE 5
#define SNR_SIZE 8

// Function to perform tanh activation on a vector
void tanhActivation(float vector[VAR2], float result[VAR2]) {
    for (int i = 0; i < VAR2; i++) {
        result[i] = tanh(vector[i]);
    }
}

void resetGate(float Wir[VAR2][VAR1], float X[VAR1], float bir[VAR2], float Wt[VAR2][VAR2], float ht[VAR2], float bt[VAR2], float r[VAR2]) {
    float r1[VAR2], r2[VAR2];
    multiplyMatrixVector(Wir, X, r1);
    multiplyMatrixVector2(Wt, ht, r2);
    for(int i = 0; i < VAR2; i++){
        r[i] = r1[i] + r2[i] + bir[i] + bt[i];
    }
}

void updateGate(float Wiz[VAR2][VAR1], float X[VAR1], float biz[VAR2], float Wt[VAR2][VAR2], float ht[VAR2], float bt[VAR2], float z[VAR2]) {
    float z1[VAR2], z2[VAR2];
    multiplyMatrixVector(Wiz, X, z1);
    multiplyMatrixVector2(Wt, ht, z2);
    for(int i = 0; i < VAR2; i++){
        z[i] = z1[i] + z2[i] + biz[i] + bt[i];
    }
}

void tanh_layer(float Win[VAR2][VAR1], float X[VAR1], float bin[VAR2], float r[VAR2], float Whn[VAR2][VAR2], float ht_1[VAR2], float bh[VAR2], float n[VAR2]) {
    float n1[VAR2], n2[VAR2];
    multiplyMatrixVector(Win, X, n1);
    multiplyMatrixVector2(Whn, ht_1, n2);
    for(int i = 0; i < VAR2; i++){
        n[i] = (n2[i] + bh[i]) * r[i];
        n[i] += (n1[i] + bin[i]);
    }
    tanhActivation(n, n);
}

// DOUBT IN THIS FUNCTION
void ht_new(float z[VAR2], float ht_1[VAR2], float n[VAR2], float ht[VAR2]) {
    for(int i = 0; i < VAR2; i++){
        ht[i] = (1 - z[i]) * n[i] + z[i] * ht_1[i];
    }
}

void linear_layer(float W[VAR4][VAR2], float n[VAR2], float b[VAR4], float y[VAR4]) {
    multiplyMatrixVector(W, n, y);
    for(int i = 0; i < VAR3; i++){
        y[i] += b[i];
    }
}

void GRU() {
    // Sample usage
    float ht[VAR1], r[VAR2];
    float z[VAR2];
    float ht_1[VAR1], n[VAR2];
    float output_gru[FRAME_SIZE][VAR4];

    // Initialize matrices and vectors
    for(int i = 0; i < VAR2; i++){
        r[i] = 0;
        z[i] = 0;
        n[i] = 0;
    }

    for(int i = 0; i < VAR1; i++){
        ht[i] = 0;
        ht_1[i] = 0;
    }

    // Call the functions
    for(int i = 0; i < FRAME_SIZE; i++){
            resetGate(rgiw, input_0[i], rgib, rghw, ht, rghb, r);
            updateGate(ugiw, input_0[i], ugib, ughw, ht, ughb, z);
            tanh_layer(tgiw, input_0[i], tgib, r, tghw, ht_1, tghb, n);
            ht_new(z, ht_1, n, ht);
            linear_layer(outWeight, n, outBias, output_gru[i]);
    }

    // Use the results as needed
    // ...

}