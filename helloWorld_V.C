#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"


int main()
{
    init_platform();

    GRU();
    cleanup_platform();
    return 0;
}
#include <stdio.h>
#include <stdlib.h>

#include "matrixMultiplication.h"
#include "hiddenGateBias.h"
#include "hiddenGateWeights.h"
#include "inputGateBias.h"
#include "inputGateWeights.h"
#include "linearLayerbias.h"
#include "linearLayerWeight.h"
#include "input_0.h"
#include "input_5.h"
#include "input_10.h"
#include "input_15.h"
#include "input_20.h"
#include "input_25.h"
#include "input_30.h"
#include "input_35.h"
#include "input_40.h"
#include "y_df_0.h"
#include "y_df_5.h"
#include "y_df_10.h"
#include "y_df_15.h"
#include "y_df_20.h"
#include "y_df_25.h"
#include "y_df_30.h"
#include "y_df_35.h"
#include "y_df_40.h"

#include "math.h"
#include <complex.h>

#define VAR_104 104
#define VAR_48 48
#define VAR_52 52
#define VAR_96 96
#define FRAME_SIZE 1
#define SNR_SIZE 8




float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}


// Function to perform tanh activation on a vector
void tanhActivation(float vector[VAR_48], float result[VAR_48]) {
    for (int i = 0; i < VAR_48; i++) {
        result[i] = tanh(vector[i]);
    }
}

void resetGate(float Wir[VAR_48][VAR_104], float X[VAR_104], float bir[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float r[VAR_48]) {
    float r1[VAR_48], r2[VAR_48];
    for(int i=0;i<VAR_48;i++){
    	r1[i] = 0;
    	r2[i] = 0;
    }
    multiplyMatrixVector(Wir, X, r1);
    multiplyMatrixVector2(Wt, ht, r2);
    for(int i = 0; i < VAR_48; i++){
        r[i] = sigmoid(r1[i] + r2[i] + bir[i] + bt[i]);
    }

}

void updateGate(float Wiz[VAR_48][VAR_104], float X[VAR_104], float biz[VAR_48], float Wt[VAR_48][VAR_48], float ht[VAR_48], float bt[VAR_48], float z[VAR_48]) {
    float z1[VAR_48], z2[VAR_48];
    multiplyMatrixVector(Wiz, X, z1);
    multiplyMatrixVector2(Wt, ht, z2);
    for(int i = 0; i < VAR_48; i++){
        z[i] = sigmoid(z1[i] + z2[i] + biz[i] + bt[i]);
    }
}

void tanh_layer(float Win[VAR_48][VAR_104], float X[VAR_104], float bin[VAR_48], float r[VAR_48], float Whn[VAR_48][VAR_48], float ht_1[VAR_48], float bh[VAR_48], float n[VAR_48]) {
    float n1[VAR_48], n2[VAR_48];
    multiplyMatrixVector(Win, X, n1);
    multiplyMatrixVector2(Whn, ht_1, n2);
    for(int i = 0; i < VAR_48; i++){
        n[i] = (n2[i] + bh[i]) * r[i];
        n[i] += (n1[i] + bin[i]);
    }
    tanhActivation(n, n);
}

// DOUBT IN THIS FUNCTION
void ht_new(float z[VAR_48], float ht_1[VAR_48], float n[VAR_48], float ht[VAR_48]) {
    for(int i = 0; i < VAR_48; i++){
        ht[i] = (1 - z[i]) * n[i] + z[i] * ht_1[i];
    }
}

void linear_layer(float W[VAR_96][VAR_48], float n[VAR_48], float b[VAR_96], float y[VAR_96]) {
    multiplyMatrixVector3(W, n, y);
    for(int i = 0; i < VAR_96; i++){
        y[i] += b[i];
    }
}

float complex complexDiv(float complex a, float complex b){
//	printf("a=%f+I%f \n", crealf(a), cimagf(a));
//	printf("b=%f+I%f \n", crealf(b), cimagf(b));
	float mag = crealf(b)*crealf(b) + cimagf(b)*cimagf(b);
//	printf("mag=%f \n", mag);
	float complex out = (crealf(a)*crealf(b) + cimagf(a)*cimagf(b) + I*(-crealf(a)*cimagf(b)+crealf(b)*cimagf(a)))/mag;
//	printf("out=%f+%fI\n\n", crealf(out), cimagf(out));
	return out;

}


//void complexDiv(float complex z1, float complex z2, float complex result) {
//    // Calculate the complex conjugate of z2
//    float complex conj_z2 = conjf(z2);
//
//    // Multiply z1 by the conjugate of z2
//    float complex numerator = z1 * conj_z2;
//
//    // Calculate the squared magnitude of z2
//    float magnitude_z2_squared = crealf(z2) * crealf(z2) + cimagf(z2) * cimagf(z2);
//
//    // Divide the real and imaginary parts of the numerator by the squared magnitude of z2
//    float real_part = crealf(numerator) / magnitude_z2_squared;
//    float imag_part = cimagf(numerator) / magnitude_z2_squared;
//
//    // Construct and return the result
//    result = real_part + imag_part * I;
////    return result;
//}

void equalizer(float complex yd_real[VAR_48], float hprev[VAR_96], int i, float complex yeq[VAR_48]) {

    for(int i=0;i<VAR_48;i++){
    	yeq[i]=complexDiv(yd_real[i],(hprev[i]+I*hprev[i+48]));
    }
}

void demapping(float complex yeq[VAR_48], float complex d[VAR_52], float complex yd[VAR_52], float complex hdpa[VAR_52]){
    int k=0;
    for(int i=0;i<VAR_52;i++){
        if(i==6 || i==20 || i==31){
            d[i] = 1;
        }
        else if(i==45){
            d[i] = -1;
        }
        else{
            if(crealf(yeq[k])>=0 && cimagf(yeq[k])>=0){
                d[i] = 1 + I;
            }
            else if(crealf(yeq[k])>=0 && cimagf(yeq[k])<=0){
                d[i] = 1 - I;
            }
            else if(crealf(yeq[k])<=0 && cimagf(yeq[k])>=0){
                d[i] = -1 + I;
            }
            else if(crealf(yeq[k])<=0 && cimagf(yeq[k])<=0){
                d[i] = -1 - I;
            }
            k++;
        }
        d[i] = d[i] * 0.707;

    }
    // ls estimation:
    for (int i=0;i<VAR_52;i++){
//    	hdpa[i] = crealf(yd[i])/crealf(d[i]) + I * cimagf(yd[i])/cimagf(d[i]);
    	hdpa[i]=complexDiv(yd[i],d[i]);
    }

}

void GRU() {
    // Sample usage
    float ht[VAR_48], r[VAR_48];
    float z[VAR_48];
    float ht_1[VAR_48], n[VAR_48], output_gru[FRAME_SIZE][VAR_96];
    float complex yd_dsym[VAR_48]; // Data symbol in 96X1
    float complex yd[VAR_48]; // Data symbol in 48X1 complex values
    float complex d[VAR_52]; // Complex data nearest to the reference symbol (Demodulated Output)
    float complex hDPA_complex[VAR_52];
    float hLS[VAR_104], hLS_D[VAR_96], hDPA[VAR_104];
    float hprev[VAR_104];
    float complex yeq[VAR_48];
    int iter;

    for(int i=0;i<VAR_48;i++){
    	ht[i]=0;
    	ht_1[i]=0;
    }

    for(int i = 0; i < VAR_104; i++){
        hLS[i]=0;
        hDPA[i] = 0;
    }
    for(int i=0;i<VAR_52;i++){
    	d[i] = 1+I;
    	yd[i] = 0+0*I;
    }

    for(int i=0;i<VAR_48;i++){
    	yd_dsym[i] = 0;
    }

    // Call the functions
    for(int i = 0; i < FRAME_SIZE; i++){

        int k = 0;
		for (int j = 0; j < VAR_104; j++) {
			if (j != 6 && j != 20 && j != 31 && j != 45 && j != 58 && j != 72 && j != 83 && j != 97) {
				hLS_D[k] = input_0[i][j];
				k++;
			}
			hLS[j] = input_0[i][j];

		}


        for(iter=0;iter<100;iter++){
    		k=0;
    		int m=0;
    		for(int j=0;j<VAR_52;j++){
    			yd[j] = y_df_0[i][iter][j];
    		}

            for(int j=0;j<VAR_52;j++){
//    			if (j != 6 && j != 20 && j != 31 && j != 45) {
//                    yd_real[k] = crealf(y_df_0[i][iter][j]);
////                    yd_real[k+48] = cimagf(y_df_0[i][iter][j]);
//                    k++;
//
//    			}
            	if(j==6 || j==20 || j==31 || j==45){
            		continue;
            	}
            	else{
            		yd_dsym[k] = crealf(y_df_0[i][iter][j]) + I*cimagf(y_df_0[i][iter][j]);
            		k++;
            	}

             }



            k=0,m=0;

            if(iter==0){
				resetGate(rgiw, hLS, rgib, rghw, ht_1, rghb, r);
				updateGate(ugiw, hLS, ugib, ughw, ht_1, ughb, z);
				tanh_layer(tgiw, hLS, tgib, r, tghw, ht_1, tghb, n);
				ht_new(z, ht_1, n, ht);
				linear_layer(outWeight, n, outBias, output_gru[i]);
				for(k=0;k<VAR_48;k++){
					ht_1[k] = ht[k];
				}
				k=0;
	            equalizer(yd_dsym, hLS_D, i, yeq); // Gives yEqualized output of size 48X1 Complex values

	            demapping(yeq, d, yd, hDPA_complex); // hDPA_complex is now updated


	            for(int j=0;j<VAR_52;j++){
	            	hDPA[j] = (float)crealf(hDPA_complex[j]);
	                hDPA[j+52] = (float)cimagf(hDPA_complex[j]);
	            } // correct output till here

	            // Time Averaging

	            for(int j=0;j<VAR_104;j++){
	            	hprev[j] = 0.5*hLS[j] + 0.5*hDPA[j];
	            }

            }
            else{
				resetGate(rgiw, hprev, rgib, rghw, ht_1, rghb, r);
				updateGate(ugiw, hprev, ugib, ughw, ht_1, ughb, z);
				tanh_layer(tgiw, hprev, tgib, r, tghw, ht_1, tghb, n);
				ht_new(z, ht_1, n, ht);
				linear_layer(outWeight, n, outBias, output_gru[i]);
				for(k=0;k<VAR_48;k++){
					ht_1[k] = ht[k];
				}
	            equalizer(yd_dsym, output_gru[i], i, yeq); // Gives yEqualized output of size 48X1 Complex values - tested and works like a charm

	            demapping(yeq, d, yd, hDPA_complex); // hDPA_complex is now updated


	            for(int j=0;j<VAR_52;j++){
	            	hDPA[j] = (float)crealf(hDPA_complex[j]);
	                hDPA[j+52] = (float)cimagf(hDPA_complex[j]);
	             }
	            // Time Averaging

	            for(int j=0;j<VAR_104;j++){
	            	hprev[j] = 0.5*hDPA[j] + 0.5*hprev[j];
	            }
	            printf("TA Done, hprev Updated");

            }

        }
        printf("All Iterations Done.");
    }
}
