#ifndef POA_H
#define POA_H

#include <unordered_map>
#include <vector>
#include <random>
#include "task.h"

typedef uint32_t kmer;
typedef short Edge;


using namespace std;

void read_batch_2(vector<vector<string>> &batch, size_t size, string filename);

vector<vector<string>> read_batch(string &filepath, int max_N, int max_W, int batch_size);

void get_bmean_batch_result(vector<vector<string>> windows, vector<vector<string>> &results);

vector<string> test_bmean(vector<string> &W, int maxMSA, string path);

void test_MSA_batch(vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &obtained, 
		    vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> &expected);

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC(vector<vector<string>> &test_in);

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_pool(vector<vector<string>> &test_in);

vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_gpu(vector<vector<string>> &test_in);
	
vector<pair<vector<vector<string>>, unordered_map<kmer, unsigned>>> testMSABMAAC_gpu_std(vector<vector<string>> &test_in);

vector<string> generate_random_window(int max_L, int min_L, int max_N);

void get_bmean_batch_result_mt(vector<vector<string>> &windows, vector<vector<string>> &results, unsigned int n_threads);

void get_bmean_batch_result_gpu(vector<vector<string>> windows, vector<vector<string>> &results, int &c, int max_s, int max_w);

void test_batch(vector<vector<string>> &obtained, vector<vector<string>> &expected);

void test_task_batch(vector< poa_gpu_utils::Task< vector< string > > > &obtained, vector<vector<string>> &expected);

vector<vector<string>> get_random_sample(int batch_size, int max_L, int min_L, int max_N, int min_N);

__global__ void assign_device_memory(int* ledges_offs, unsigned char* lletters, Edge* ledges, int* ebounds, unsigned char* ennodes, unsigned char* seq_ids, unsigned char* nletters, Edge* nedges, int* nedgebounds, unsigned char* n_end_nodes, unsigned char* n_seq_ids, unsigned char* dletters, Edge* dedges, int* dedgebounds, unsigned char* d_end_nodes, unsigned char* d_seq_ids, unsigned char* moves, short* diagonals_sc, short* diagonals_gx, short* diagonals_gy, int* d_offs, int* xy, int* yx, int* oldlg, int* dynlg, const int num_blocks);

 __global__ void init_diagonals(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets);

 __global__ void sw_align(int i_seq_idx, int j_seq_idx, int max_gapl, int uses_global, int* nseq_offsets); 
	
 __device__ void trace_back_lpo_alignment(int len_x, int len_y, unsigned char* move_x, unsigned char* move_y, Edge* x_left, Edge* y_left, int* start_x, int* start_y, int best_x, int best_y, int* x_to_y, int* y_to_x, int* d_offsets);

 __global__ void compute_d_offsets(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

 __global__ void compute_new_lpo_size(int i_seq_idx, int j_seq_idx, int* nseq_offsets, int* space_exceeded);

 __global__ void fuse_lpo(int i_seq_idx, int j_seq_idx, int* nseq_offsets);

 __global__ void copy_new_lpo_data(int j_seq_idx, int* nseq_offsets);

 __global__ void compute_edge_offsets(int* seq_offsets, int* nseq_offsets);

 __global__ void generate_lpo(char* seq, int* seq_offsets, int* nseq_offsets, int seq_idx);

 __global__ void copy_result_sizes(int *nseq_offsets, int* res_size);

 __global__ void compute_result(int *nseq_offsets, char* result, int* seq_offsets, int seq_idx);

 __global__ void suffix_sum(int* d_ptr, const int num_blocks);

 #endif