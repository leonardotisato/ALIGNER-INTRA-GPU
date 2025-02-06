#include <unordered_map>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <map>
#include "../include/poagpu.cuh"


void get_bmean_batch_result_gpu(vector<vector<string>> windows, vector<vector<string>> &results, int &c /*, int max_s, int max_w*/){

	poa_gpu_utils::TaskRefs T;		// struct con dentro TUUUUUTTO
	size_t size = windows.size();

	// vector<vector<string>> result_GPU;
	vector<poa_gpu_utils::Task<vector<string>>> gpu_tasks(size, poa_gpu_utils::Task<vector<string>>(0,0,vector<string>()));
	vector<poa_gpu_utils::Task<vector<string>>> gpu_res(size, poa_gpu_utils::Task<vector<string>>(0,0,vector<string>()));
	
	//task transfer
	int i = 0;
	for(auto s : windows){
		poa_gpu_utils::Task<vector<string>> t = poa_gpu_utils::Task<vector<string>>(i, i, s);
		gpu_tasks[i] = t;
		i++;
	}

	// TTy = poa_gpu_utils::get_task_type_direct(max_w, max_s);

	auto start = NOW;

	// poa_gpu_utils::sel_gpu_POA_alloc(T, TTy);
	gpu_POA_alloc(T);
	// poa_gpu_utils::sel_gpu_POA(gpu_tasks, T, gpu_res, 0, TTy);
	gpu_POA(gpu_tasks, T, gpu_res, 0);
	gpu_POA_free(T);

	auto end = NOW;
	c = duration_cast<microseconds>(end - start).count();
	std::cout << "Duration: " << c << " microseconds" << std::endl;

	for(auto r : gpu_res){
		results.push_back(r.task_data);
	}
}


vector<string> generate_random_window(int max_L, int min_L, int min_N, int max_N) {

	// random_device rd;
	// default_random_engine generator(rd());
	
	// uniform_int_distribution<int> L_distribution(min_L, max_L);
	// uniform_int_distribution<int> N_distribution(min_N, max_N);	
	// uniform_int_distribution<int> char_distribution(0,3);

	// int L = L_distribution(generator);
	
	// vector<string> window;

	// for(int i = 0; i < L; i++){

	// 	int N = N_distribution(generator);
	// 	string sequence = "";
	
	// 	for(int j = 0; j < N; j++){
	// 		char c = char_map[char_distribution(generator)];
	// 		sequence += c;
			
	// 	}
	// 	window.push_back(sequence);
	// }

	vector<string> window;
    window.push_back("AG");
    window.push_back("ACTGA");
	window.push_back("TTC");
	window.push_back("TTC");

	return window;

}

vector<vector<string>> get_random_sample(int batch_size, int max_L = MAX_L, int min_L = MIN_L, int max_N = MAX_N, int min_N = MIN_N) { 		
	std::cout << "Alive!\n";
	vector<vector<string>> sample;	
	// int step = batch_size / 10;
	// int perc = 0;

	cout << "Sample generation: size=" << batch_size << ", L=[" << min_L << "," << max_L << "], N=[" << min_N << "," << max_N << "]\n";

	for(int i = 0; i < batch_size; i++) {
// PERCHE' NON EMPLACE_BACK ?? -------------------------------------------------------------------------------------------------
		sample.push_back(generate_random_window(max_L, min_L, min_N, max_N));
		// if(i % step == 0){ cout << "Generation: [" << perc << "%]\n"; perc += 10;  }
	}
	return sample;
}