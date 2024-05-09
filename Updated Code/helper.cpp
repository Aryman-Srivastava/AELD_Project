#include "helper.h"
#include "/home/varun/AELD_Project_IP/tanh_lut.h"
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

//float sigmoid(float x) {
//    return 1.0 / (1.0 + exp(-x));
//}

// Define the piecewise linear function
float sigmoid(float x) {
    float parameters[][3] = {
    {0.0078125, 0.05, -3.4},
    {0.0625, 0.24, -1.3},
    {0.25, 0.5, 1.3},
    {0.0625, 0.76, 3.4},
    {0.0078125, 0.95, INFINITY}};
    int i;
    for (i = 0; i < sizeof(parameters) / sizeof(parameters[0]); i++) {
        double m = parameters[i][0];
        double k = parameters[i][1];
        double n = parameters[i][2];
        if (x <= n) {
            return m * x + k;
        }
    }
    // Return 0 for values of x greater than the last interval
    return 0.0;
}
//#define LUT_RESOLUTION 1000
//#define LUT_RANGE 10.0
//
//// Define the global Look-Up Table
//static float lut_x[LUT_RESOLUTION];
//static float lut_y[LUT_RESOLUTION];
//
//void initialize_lut() {
//        for (int i = 0; i < LUT_RESOLUTION; ++i) {
//            lut_x[i] = -LUT_RANGE + (2 * LUT_RANGE / (LUT_RESOLUTION - 1)) * i;
//            lut_y[i] = tanh(lut_x[i]);
//        }
//}

float tanh_lut(float x) {
    // Initialize the LUT if not already initialized
//    initialize_lut();

    // Interpolate the value from the LUT
    int idx = (int) ((x + LUT_RANGE) / (2 * LUT_RANGE) * (LUT_RESOLUTION - 1));
    if (idx < 0)
        idx = 0;
    else if (idx >= LUT_RESOLUTION)
        idx = LUT_RESOLUTION - 1;

    return LUT_Y[idx];
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
//	initialize_lut();

    tanh_gate_loop: for(int i = 0; i < VAR_48; i++){
	#pragma HLS PIPELINE
        n_int[i] = (n2[i] + bh[i]) * r[i] + (n1[i] + bin[i]);
        n[i] = tanh_lut(n_int[i]);
    }
}

void ht_new(float z[VAR_48], float ht_1[VAR_48], float n[VAR_48], float ht[VAR_48]) {
    ht_new_loop: for(int i = 0; i < VAR_48; i++){
	#pragma HLS PIPELINE
        ht[i] = (1 - z[i]) * n[i] + z[i] * ht_1[i];
//        printf("ht = %f, z = %f, n =  %f\n", ht[i], z[i], n[i]);
    }
}

void multiplyMatrixVector_96_48_48_1(float matrix[VAR_96][VAR_48], float vector[VAR_48], float result[VAR_96]) {
    for (int i = 0; i < VAR_96; i++) {
        result[i] = 0;
		#pragma HLS PIPELINE

        for (int j = 0; j < VAR_48; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}
void linear_layer(float W[VAR_96][VAR_48], float n[VAR_48], float b[VAR_96], float y[VAR_96]) {
	multiplyMatrixVector_96_48_48_1(W, n, y);
    for(int i = 0; i < VAR_96; i++){
#pragma HLS PIPELINE
        y[i] += b[i];
    }
}

