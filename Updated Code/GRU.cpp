#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "helper.h"
#include "weights.h"

void GRU(hls::stream<axis_data> &hls, hls::stream<axis_data> &output_data){
	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis register both port=output_data
	#pragma HLS INTERFACE axis register both port=hls
//	#pragma HLS ARRAY_PARTITION variable=ugiw block factor=104 dim=2
//	#pragma HLS ARRAY_PARTITION variable=rgiw block factor=104 dim=2
//	#pragma HLS ARRAY_PARTITION variable=tgiw block factor=104 dim=2
//	#pragma HLS ARRAY_PARTITION variable=rghw block factor=48 dim=2
//	#pragma HLS ARRAY_PARTITION variable=ughw block factor=48 dim=2
//	#pragma HLS ARRAY_PARTITION variable=tghw block factor=48 dim=2
//	#pragma HLS ARRAY_PARTITION variable=rghb block factor=48 dim=1
//	#pragma HLS ARRAY_PARTITION variable=ughb block factor=48 dim=1
//	#pragma HLS ARRAY_PARTITION variable=tghb block factor=48 dim=1
//	#pragma HLS ARRAY_PARTITION variable=rgib block factor=48 dim=1
//	#pragma HLS ARRAY_PARTITION variable=ugib block factor=48 dim=1
//	#pragma HLS ARRAY_PARTITION variable=tgib block factor=48 dim=1

//	#pragma HLS ARRAY_PARTITION variable=outWeight block factor=48 dim=2

	float hls_data[VAR_104], ht[VAR_48];
//	#pragma HLS ARRAY_PARTITION variable=hls_data complete dim=1

	float GRU_output[VAR_48];
//	#pragma HLS ARRAY_PARTITION variable=GRU_output complete dim=1
	float ll_output[VAR_96];

	float r[VAR_48], z[VAR_48], n[VAR_48];
//	#pragma HLS ARRAY_PARTITION variable=n block factor=48 dim=1
//	#pragma HLS ARRAY_PARTITION variable=z block factor=48 dim=1
//	#pragma HLS ARRAY_PARTITION variable=r block factor=48 dim=1
	axis_data local_stream;
	int i;
	streamloop1: for(i=0;i<VAR_104;i++){
	#pragma HLS PIPELINE
		local_stream = hls.read();
		float temp = local_stream.data;
		hls_data[i] = temp;
	}

	streamloop2: for(i=0;i<VAR_48;i++){
	#pragma HLS PIPELINE
		local_stream = hls.read();
		ht[i] = local_stream.data;
	}

	resetGate(rgiw, hls_data, rgib, rghw, ht, rghb, r);
	updateGate(ugiw, hls_data, ugib, ughw, ht, ughb, z);
	tanh_layer(tgiw, hls_data, tgib, r, tghw, ht, tghb, n);
	ht_new(z, ht, n, GRU_output);
	linear_layer(outWeight, GRU_output, outBias, ll_output);

	streamloop3: for(i=0;i<VAR_96;i++){
	#pragma HLS PIPELINE
		local_stream.data = ll_output[i];
		local_stream.last = 0;
		output_data.write(local_stream);
	}
	streamloop4: for(i=0;i<VAR_48;i++){
	#pragma HLS PIPELINE

		local_stream.data = GRU_output[i];
		if(i == VAR_48-1){
			local_stream.last = 1;
		}
		else{
			local_stream.last = 0;
		}
		output_data.write(local_stream);
	}
}
