#include <math.h>
#include <stdio.h>
#include <hls_stream.h>
#include <ap_int.h>

#define VAR_104 104
#define VAR_48 48
#define VAR_52 52
#define VAR_96 96
#define solution1

typedef float Mat_Dtype;

struct axis_data{
	Mat_Dtype data;
	ap_uint<1> last;
};

void multiplyMatrixVector_48_104(float matrix[VAR_48][VAR_104], float vector[VAR_104], float result[VAR_48]);
void multiplyMatrixVector_48_48(float matrix[VAR_48][VAR_48], float vector[VAR_48], float result[VAR_48]);
void multiplyMatrixVector_96_48(float matrix[VAR_96][VAR_48], float vector[VAR_48], float result[VAR_48]);
float sigmoid(float x);
void resetGate(float Wir[VAR_48][VAR_104], float X[VAR_104], float bir[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float r[VAR_48]);
void updateGate(float Wiz[VAR_48][VAR_104], float X[VAR_104], float biz[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float z[VAR_48]);
void tanh_layer(float Win[VAR_48][VAR_104], float X[VAR_104], float bin[VAR_48], float r[VAR_48], float Whn[VAR_48][VAR_48], float ht_1[VAR_48], float bh[VAR_48], float n[VAR_48]);
void ht_new(float z[VAR_48], float ht_1[VAR_48], float n[VAR_48], float ht[VAR_48]);
void linear_layer(float W[VAR_96][VAR_48], float n[VAR_48], float b[VAR_96], float y[VAR_96]);
