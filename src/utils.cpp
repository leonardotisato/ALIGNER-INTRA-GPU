#include <unordered_map>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <map>
#include "../include/poagpu.cuh"


int check_input_int(string &arg){
	
	try{
		size_t pos;
		int arg_i = stoi(arg, &pos);
		if(pos < arg.size()){
			std::cerr << "Trailing characters after number: " << arg << '\n';
		}
		return arg_i;
	} catch (invalid_argument const &ex) {
		std::cerr << "Invalid number: " << arg << '\n';
		return -1;
	} catch (out_of_range const &ex) {
		std::cerr << "Number out of range: " << arg << '\n';
		return -1;
	}
	
}


void read_batch(vector<vector<string>> &reads, size_t size, string filename){

	ifstream infile(filename);
    int i = 0;

    if (!infile.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        return;
    }

    string line;
    vector<string> readsVector;

    // Lettura del file FASTA o simile
    while (getline(infile, line)) {
        // Rimozione spazi bianchi iniziali e finali
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        if (!line.empty()) {
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
        }

        if (line.empty() || line[0] == '>') continue;

        readsVector.push_back(line);         

		i++;
		if(i % size == 0) {
			reads.push_back(readsVector);
			readsVector.clear();
		}
    }

	if (!readsVector.empty()) {
        reads.push_back(readsVector);
    }

    infile.close();
}

void print_reads(vector<vector<string>> reads) {
	
	for(int i = 0; i < reads.size(); i++) {
		cout << "Batch " << i << endl;
		for(int j = 0; j < reads[i].size(); j++) {
			cout << reads[i][j] << endl;
		}
	}
}


void get_bmean_batch_result_gpu(vector<vector<string>> reads, int &c){

	poa_gpu_utils::TaskRefs T;		// struct con dentro TUUUUUTTO
	// size_t size = reads.size();

	auto start = NOW;

	gpu_POA_alloc(T);
	gpu_POA(reads, T);
	gpu_POA_free(T);

	auto end = NOW;
	c = duration_cast<microseconds>(end - start).count();
	std::cout << "Duration: " << c << " microseconds" << std::endl;
}