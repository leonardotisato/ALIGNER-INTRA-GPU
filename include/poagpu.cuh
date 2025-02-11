#ifndef POAGPU_H
#define POAGPU_H

#include <cuda_runtime.h>
#include <chrono>
#include <unistd.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <numeric>
#include <stdexcept>
#include <thread>
#include "poa.h"

#define DEBUG 1


#define SL 32
#define MAXL 10
#define	WL 10
#define BDIM 10

#define EDGE_F 3 // Heuristic mean degree for graphs
#define N_THREADS 64


using namespace std;
using namespace chrono;
using namespace poa_gpu_utils;

#define NOW high_resolution_clock::now() 

#define cudaErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {

	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

static const int NOT_ALIGNED = -1;

void init_kernel_block_parameters(vector<Task<vector<string>>> &window_batch, char** sequences, vector<int> &nseq_offsets, vector<int> &seq_offsets, int* tot_nseq, int first_el) {
	
	int batch_size = window_batch.size();						
	int n = (first_el + BDIM < batch_size) ? BDIM : batch_size - first_el;
	for(int window_idx = first_el; window_idx < first_el + BDIM && window_idx < batch_size; window_idx++) {
		
		nseq_offsets[window_idx - first_el] = window_batch[window_idx].task_data.size();
	}
	partial_sum(nseq_offsets.begin(),nseq_offsets.end(),nseq_offsets.begin());
	*tot_nseq = nseq_offsets[n-1];
	seq_offsets = vector<int>(*tot_nseq);
	
	int sequence_idx = 0;
	for(int window_idx = first_el; window_idx < first_el + BDIM && window_idx < window_batch.size(); window_idx++) {
		vector<string> &window = window_batch[window_idx].task_data;			
		int wsize = window.size();
		for(int i = 0; i < wsize; i++) {
			seq_offsets[sequence_idx] = window[i].size();
			sequence_idx++;
		}
	}
	partial_sum(seq_offsets.begin(), seq_offsets.end(), seq_offsets.begin());
	int tot_size = seq_offsets[sequence_idx-1];
	*sequences = (char*)malloc(tot_size);
	sequence_idx = 0;

	for(int window_idx = first_el; window_idx < first_el + BDIM && window_idx < window_batch.size(); window_idx++) {
			
		vector<string>& window = window_batch[window_idx].task_data;	
		for(auto seq : window) {
			int offset;
			if(sequence_idx == 0){
				offset = 0;
			}else{
				offset = seq_offsets[sequence_idx-1];
			}
			char* seq_ptr = (*sequences) + offset;
			memcpy(seq_ptr, seq.c_str(), seq.size());
			sequence_idx++;
		}
	}	
}

inline void gpu_POA_alloc(TaskRefs &T){

	T.result = (char*)malloc(WL * MAXL * BDIM);
	T.res_size = (int*)malloc(BDIM * sizeof(int));

	cudaErrchk(cudaMalloc(&T.space_exceeded, sizeof(int)));

	cudaErrchk(cudaMalloc(&T.old_len_global_d, (unsigned long long)BDIM * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.dyn_len_global_d, (unsigned long long)BDIM * sizeof(int)));

	cudaErrchk(cudaMalloc(&T.sequences_d, (unsigned long long)SL * WL * BDIM)); 
	cudaErrchk(cudaMalloc(&T.seq_offsets_d, (unsigned long long)BDIM * WL * sizeof(int))); 
	cudaErrchk(cudaMalloc(&T.nseq_offsets_d, (unsigned long long)BDIM * sizeof(int))); 
	cudaErrchk(cudaMalloc(&T.result_d, (unsigned long long)MAXL * WL * BDIM)); 
	cudaErrchk(cudaMalloc(&T.res_size_d, (unsigned long long)BDIM * sizeof(int)));

	cudaErrchk(cudaMalloc(&T.lpo_edge_offsets_d, (unsigned long long)WL * BDIM * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.lpo_letters_d, (unsigned long long)WL * SL * BDIM));
	cudaErrchk(cudaMalloc(&T.lpo_edges_d, (unsigned long long)WL * SL * EDGE_F * BDIM * sizeof(Edge)));
	cudaErrchk(cudaMalloc(&T.edge_bounds_d, (unsigned long long)WL * (SL+1) * BDIM * sizeof(int)));	
	cudaErrchk(cudaMalloc(&T.end_nodes_d, (unsigned long long)WL * SL * BDIM));
	cudaErrchk(cudaMalloc(&T.sequence_ids_d, (unsigned long long)WL * WL * SL * BDIM));
	
	cudaErrchk(cudaMalloc(&T.new_letters_global_d, (unsigned long long)MAXL * BDIM));
	cudaErrchk(cudaMalloc(&T.new_edges_global_d, (unsigned long long)MAXL * EDGE_F * BDIM * sizeof(Edge)));
	cudaErrchk(cudaMalloc(&T.new_edge_bounds_global_d, (unsigned long long)(MAXL+1) * BDIM * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.new_end_nodes_global_d, (unsigned long long)MAXL * BDIM));
	cudaErrchk(cudaMalloc(&T.new_sequence_ids_global_d, (unsigned long long)WL * MAXL * BDIM));

	cudaErrchk(cudaMalloc(&T.dyn_letters_global_d, (unsigned long long)MAXL * BDIM));
	cudaErrchk(cudaMalloc(&T.dyn_edges_global_d, (unsigned long long)MAXL * EDGE_F * BDIM * sizeof(Edge)));
	cudaErrchk(cudaMalloc(&T.dyn_edge_bounds_global_d, (unsigned long long)(MAXL+1) * BDIM * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.dyn_end_nodes_global_d, (unsigned long long)MAXL * BDIM));
	cudaErrchk(cudaMalloc(&T.dyn_sequence_ids_global_d, (unsigned long long)WL * MAXL * BDIM));

	cudaErrchk(cudaMalloc(&T.moves_global_d, (unsigned long long)2 * (MAXL+1) * (SL+1) * BDIM * sizeof(unsigned char)));
	cudaErrchk(cudaMalloc(&T.diagonals_global_sc_d, (unsigned long long)(MAXL+1)*(SL+1) * BDIM * sizeof(short)));
	cudaErrchk(cudaMalloc(&T.diagonals_global_gx_d, (unsigned long long)(MAXL+1)*(SL+1) * BDIM * sizeof(short)));
	cudaErrchk(cudaMalloc(&T.diagonals_global_gy_d, (unsigned long long)(MAXL+1)*(SL+1) * BDIM * sizeof(short)));
	cudaErrchk(cudaMalloc(&T.d_offsets_global_d, (unsigned long long)(MAXL + SL+1) * BDIM * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.x_to_ys_d, (unsigned long long)MAXL * BDIM * sizeof(int)));
	cudaErrchk(cudaMalloc(&T.y_to_xs_d, (unsigned long long)MAXL * BDIM * sizeof(int)));
	//cout << "Alloc completed\n";
}

inline void gpu_POA_free(TaskRefs &T){

	cudaDeviceReset();
	
	free(T.result);
	free(T.res_size);

}

void gpu_POA(vector<Task<vector<string>>> &input, TaskRefs &T, vector<Task<vector<string>>> &result_GPU, int res_gpu_offs) {
	
	int input_size = input.size(); // prende il numero di task che è == numero di window
	int N_BL = (input_size - 1) / BDIM + 1; // variabile dal dubbio significato (credo numero di batch) = ceil(input_size / BDIM)
	int LAST_BATCH_SIZE = (input_size - 1) % BDIM + 1; 
	int *space_exceeded = (int*)malloc(sizeof(int));

	vector<vector<string>> result_data(input_size);


	T.nseq_offsets = vector<int>(BDIM);

	assign_device_memory<<<1, 1>>>(T.lpo_edge_offsets_d, T.lpo_letters_d, T.lpo_edges_d, 
				       T.edge_bounds_d, T.end_nodes_d, T.sequence_ids_d, 
				       T.new_letters_global_d, T.new_edges_global_d, T.new_edge_bounds_global_d, 
				       T.new_end_nodes_global_d, T.new_sequence_ids_global_d, 
				       T.dyn_letters_global_d, T.dyn_edges_global_d, T.dyn_edge_bounds_global_d, 
				       T.dyn_end_nodes_global_d, T.dyn_sequence_ids_global_d,
				       T.moves_global_d, T.diagonals_global_sc_d, T.diagonals_global_gx_d, T.diagonals_global_gy_d, 
                                       T.d_offsets_global_d, T.x_to_ys_d, T.y_to_xs_d, T.old_len_global_d, T.dyn_len_global_d, BDIM);

	cudaStreamSynchronize(0);

	for(int b = 0; b < N_BL; b++){

		cout << b << "th iteration" << endl;

		int block_offset = b * BDIM;
		int BLOCKS;
		if(b == N_BL-1){
			BLOCKS = LAST_BATCH_SIZE;
		}else{
			BLOCKS = BDIM;
		}

		init_kernel_block_parameters(input, &T.sequences, T.nseq_offsets, T.seq_offsets, &T.tot_nseq, block_offset);
		
		//cout << "Start memcpy\n";
		//cout << "Memcpy of " << T.seq_offsets[T.tot_nseq-1] << " bytes\n";
		
		cudaErrchk(cudaMemcpy(T.space_exceeded, space_exceeded, sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(T.sequences_d, T.sequences, T.seq_offsets[T.tot_nseq-1], cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(T.seq_offsets_d, T.seq_offsets.data(), T.tot_nseq * sizeof(int), cudaMemcpyHostToDevice));
		cudaErrchk(cudaMemcpy(T.nseq_offsets_d, T.nseq_offsets.data(), (unsigned long long)BLOCKS * sizeof(int), cudaMemcpyHostToDevice));
		
		//cout << "Compute edge offsets\n";

		compute_edge_offsets<<<BLOCKS, WL>>>(T.seq_offsets_d, T.nseq_offsets_d);
		
		cudaStreamSynchronize(0);
		
		//cout << "Generate LPO\n";

		for(int i = 0; i < WL; i++){
			generate_lpo<<<BLOCKS, SL+1>>>(T.sequences_d, T.seq_offsets_d, T.nseq_offsets_d, i);
		}

		// print LPO
		// for (int b = 0; b < BLOCKS; ++b) {
		// 	int block_offset = (b == 0) ? 0 : T.nseq_offsets[b - 1];
		// 	int seq_len = (block_offset + i == 0) ? T.seq_offsets[block_offset + i] :
		// 				T.seq_offsets[block_offset + i] - T.seq_offsets[block_offset + i - 1];

		// 	std::cout << "Block " << b << ", Sequence " << i << ": ";
		// 	for (int i = 0; i < seq_len; ++i) {
		// 		std::cout << T.sequences[block_offset + i];
		// 	}
		// 	std::cout << std::endl;
		// }

		
		int i_seq_idx = 0;
		
		printf("\n\nGRAPH CREATED --> BEGIN ALIGNMENT\n");

		cudaStreamSynchronize(0);

		// parametro 3 da sostituire !!!
				
		compute_d_offsets<<<BLOCKS, 1>>>(i_seq_idx, 3, T.nseq_offsets_d);
		
		cudaStreamSynchronize(0); 
		
		init_diagonals<<<BLOCKS,1>>>(i_seq_idx, 3, T.max_gapl, T.uses_global, T.nseq_offsets_d);
		
		//cout << "Alignment kernel call\n";
		
		sw_align<<<BLOCKS, SL+1>>>(i_seq_idx, 3, T.max_gapl, T.uses_global, T.nseq_offsets_d);
		
		cudaStreamSynchronize(0);

    }
}


#endif