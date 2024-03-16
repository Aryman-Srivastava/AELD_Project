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

#define VAR1 104
#define VAR2 48
#define VAR3 52
#define VAR4 96
#define FRAME_SIZE 1
#define SNR_SIZE 8

// Function to perform tanh activation on a vector
void tanhActivation(float vector[VAR2], float result[VAR2]) {
    for (int i = 0; i < VAR2; i++) {
        result[i] = tanh(vector[i]);
    }
}

void resetGate(float Wir[VAR2][VAR1], float X[VAR1], float bir[VAR2], float Wt[VAR2][VAR2], float ht[VAR2], float bt[VAR2], float r[VAR2]) {
    float r1[VAR2], r2[VAR2];
    for(int i=0;i<VAR2;i++){
    	r1[i] = 0;
    	r2[i] = 0;
    }
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
    multiplyMatrixVector3(W, n, y);
    for(int i = 0; i < VAR4; i++){
        y[i] += b[i];
    }
}

void equalizer(float yd_real [VAR4], float hprev[VAR4], int i, float complex yeq[VAR2]) {
    for(int i=0;i<VAR4;i++){
        yeq[i] = yd_real[i]/hprev[i] + I * yd_real[i]/hprev[i+1];
    }
}


void demapping(float complex yeq[VAR2], float complex d[VAR3]){
    int k=0;
    for(int i=0;i<VAR3;i++){
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

}

void estimate_LS(float complex yd[VAR2], float complex d[VAR3], float complex hdpa[VAR2]){
    for(int i=0;i<VAR2;i++){
        hdpa[i] = crealf(yd[i]) / crealf(d[i]) + I * cimagf(yd[i]) / cimagf(d[i]);
    }
}

void GRU() {
    // Sample usage
    float ht[VAR2], r[VAR2];
    float z[VAR2];
    float ht_1[VAR2], n[VAR2], output_gru[FRAME_SIZE][VAR4];
    float yd_real[VAR4]; // Data symbol in 96X1
    float complex yd[VAR3]; // Data symbol in 48X1 complex values
    float complex d[VAR3]; // Complex data nearest to the reference symbol (Demodulated Output)
    float complex hDPA_complex[VAR3];
    float hLS[VAR1], hLS_D[VAR4], hDPA[VAR1];
    float hprev[VAR1];
    float complex yeq[VAR2];
    int iter;

    for(int i=0;i<VAR2;i++){
    	ht[i]=0;
    	ht_1[i]=0;
    }

    for(int i = 0; i < VAR1; i++){
        hLS[i]=0;
        hDPA[i] = 0;
    }


    // Call the functions
    for(int i = 0; i < FRAME_SIZE; i++){

        int k = 0;
		for (int j = 0; j < VAR1; j++) {
			if (j != 12 && j != 40 && j != 62 && j != 90 && j != 13 && j != 41 && j != 63 && j != 91) {
				hLS_D[k] = input_0[i][j];
				k++;
			}
			hLS[j] = input_0[i][j];

		}


        for(iter=0;iter<100;iter++){
        	// Data symbol extraction in 2 formats 96X1 and 48X1 (complex) from data frame file
    		k=0;
    		int m=0;

            for(int j=0;j<VAR3;j++){
    			if (j != 6 && j != 20 && j != 31 && j != 45) {
                    yd_real[k] = (float)crealf(y_df_0[i][iter][j]);
                    yd_real[k+1] = (float)cimagf(y_df_0[i][iter][j]);
                    k+=2;
    			}
                yd[m] = y_df_0[i][iter][j];
                m++;
             }
            k=0,m=0;

            if(iter==0){
				resetGate(rgiw, hLS, rgib, rghw, ht_1, rghb, r);
				updateGate(ugiw, hLS, ugib, ughw, ht_1, ughb, z);
				tanh_layer(tgiw, hLS, tgib, r, tghw, ht_1, tghb, n);
				ht_new(z, ht_1, n, ht);
				linear_layer(outWeight, n, outBias, output_gru[i]);
				for(k=0;k<VAR2;k++){
					ht_1[k] = ht[k];
				}
	            equalizer(yd_real, hLS , i, yeq); // Gives yEqualized output of size 48X1 Complex values

	            demapping(yeq, d); // hDPA_complex is now updated

                estimate_LS(yd, hDPA_complex);

	            k=0;
	            for(int j=0;j<VAR3;j++){
	            	hDPA[k] = (float)crealf(hDPA_complex[j]);
	                hDPA[k+1] = (float)cimagf(hDPA_complex[j]);
	                k+=2;
	            }

	            // Time Averaging

	            for(int j=0;j<VAR1;j++){
	            	hprev[j] = 0.5*hLS[j] + 0.5*hDPA[j];
	            }


            }
            else{
				resetGate(rgiw, hprev, rgib, rghw, ht_1, rghb, r);
				updateGate(ugiw, hprev, ugib, ughw, ht_1, ughb, z);
				tanh_layer(tgiw, hprev, tgib, r, tghw, ht_1, tghb, n);
				ht_new(z, ht_1, n, ht);
				linear_layer(outWeight, n, outBias, output_gru[i]);
				for(k=0;k<VAR2;k++){
					ht_1[k] = ht[k];
				}
	            equalizer(yd_real, output_gru[i], i, yeq); // Gives yEqualized output of size 48X1 Complex values

	            demapping(yeq, d); // hDPA_complex is now updated

                estimate_LS(yd, d, hDPA_complex);

	            k=0;
	            for(int j=0;j<VAR3;j++){
	            	hDPA[k] = (float)crealf(hDPA_complex[j]);
	                hDPA[k+1] = (float)cimagf(hDPA_complex[j]);
	                k+=2;
	             }
	            // Time Averaging

	            for(int j=0;j<VAR1;j++){
	            	hprev[j] = 0.5*hDPA[j] + 0.5*hprev[j];
	            }
	            printf("TA Done, hprev Updated");

            }


        }
        printf("All Iterations Done.");
    }
}
