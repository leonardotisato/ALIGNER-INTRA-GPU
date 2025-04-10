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

void print_graph_input(const poa_gpu_utils::TaskRefs& T) {
    std::cout << "g_letters (" << T.num_letters << "): ";
    for (size_t i = 0; i < T.num_letters; ++i) {
        std::cout << T.g_letters[i];
    }
    std::cout << "\n";

    std::cout << "g_edges (" << T.num_edges << "): ";
    for (size_t i = 0; i < T.num_edges; ++i) {
        std::cout << T.g_edges[i] << " ";
    }
    std::cout << "\n";

    std::cout << "g_offsets (" << T.num_offsets << "): ";
    for (size_t i = 0; i < T.num_offsets; ++i) {
        std::cout << T.g_offsets[i] << " ";
    }
    std::cout << "\n";
}

void print_reads(vector<vector<string>> reads) {

	cout << "Printing reads: "<< endl;
	for(int i = 0; i < reads.size(); i++) {
		cout << i << endl;
		cout << "Batch " << i << endl;
		for(int j = 0; j < reads[i].size(); j++) {
			cout << reads[i][j] << endl;
		}
	}
}


void read_batch(vector<vector<string>> &reads, size_t size, string filename){

	ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Errore: Impossibile aprire il file " << filename << endl;
        return;
    }

    string line;
    vector<string> allReads;

    // Legge l'intero file e raccoglie le righe utili (non intestazioni, non vuote)
    while (getline(infile, line)) {
        // Rimozione spazi bianchi iniziali e finali
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        if (!line.empty())
            line.erase(line.find_last_not_of(" \t\n\r") + 1);

        // Salta righe vuote o linee di intestazione (che iniziano con '>')
        if (line.empty() || line[0] == '>')
            continue;
        allReads.push_back(line);
    }
    infile.close();

    // Calcola il numero totale di reads e la dimensione base per ogni gruppo
    int total = allReads.size();
    int base = total / size;
    int remainder = total % size; // i primi "remainder" gruppi avranno 1 elemento in più

    int index = 0;
    for (size_t group = 0; group < size; group++) {
        // Per ogni gruppo, calcola quanti elementi deve avere
        int groupSize = base + (group < remainder ? 1 : 0);
        vector<string> groupReads;
        for (int j = 0; j < groupSize; j++) {
            groupReads.push_back(allReads[index++]);
        }
        reads.push_back(groupReads);
    }

    // cout << "Distribuzione effettuata: " << endl;
    // for (size_t i = 0; i < reads.size(); i++) {
    //     cout << "Gruppo " << i << " ha " << reads[i].size() << " reads" << endl;
    // }
}


void skip_empty_and_label(std::ifstream& file, std::string& line) {
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '>') break;
    }
}

template<typename T>
T* parse_array_from_line(const std::string& line, size_t& count) {
    count = 0;
    const char* ptr = line.c_str();
    while (*ptr) {
        if (std::isdigit(*ptr) || (*ptr == '-' && std::isdigit(*(ptr+1)))) {
            ++count;
            while (*ptr && *ptr != ' ') ++ptr;
        }
        while (*ptr == ' ') ++ptr;
    }

    T* array = new T[count];
    ptr = line.c_str();
    size_t i = 0;
    while (*ptr) {
        if (std::isdigit(*ptr) || (*ptr == '-' && std::isdigit(*(ptr+1)))) {
            array[i++] = static_cast<T>(std::atoi(ptr));
            while (*ptr && *ptr != ' ') ++ptr;
        }
        while (*ptr == ' ') ++ptr;
    }

    return array;
}

void read_graph_input(std::string filename, poa_gpu_utils::TaskRefs &T) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore apertura file: " << filename << std::endl;
        return;
    }

    std::string line;

    // g_letters (char*)
    skip_empty_and_label(file, line);
    T.num_letters = line.size();
    T.g_letters = new char[T.num_letters + 1];
    std::strcpy(T.g_letters, line.c_str());

    // g_edges (Edge*)
    skip_empty_and_label(file, line);
    T.g_edges = parse_array_from_line<Edge>(line, T.num_edges);

    // g_offsets (int*)
    skip_empty_and_label(file, line);
    T.g_offsets = parse_array_from_line<int>(line, T.num_offsets);

    file.close();
}



void get_bmean_batch_result_gpu(vector<vector<string>> reads, int &c, const int numBlocks, int batchSize, char* graph_path){

	poa_gpu_utils::TaskRefs T;		// struct con dentro TUUUUUTTO
	// size_t size = reads.size();

    string filepath(graph_path);

    read_graph_input(filepath, T);
    print_graph_input(T);

	auto start = NOW;

	gpu_POA_alloc(T, numBlocks, batchSize);
	gpu_POA(reads, T, numBlocks, batchSize);
	gpu_POA_free(T);

	auto end = NOW;
	c = duration_cast<microseconds>(end - start).count();
	std::cout << "Duration: " << c << " microseconds" << std::endl;
}