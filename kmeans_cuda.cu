// File: kmeans_cuda.cu
// Descrição: K-means adaptado para execução em GPU via CUDA com saída de clusters

#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

// ─────────── CONFIGURAÇÃO ───────────
#define DATA_FILE       "covtype.csv"    // Arquivo CSV de entrada
#define DEFAULT_K       10               // Número de clusters
#define DEFAULT_MAX_IT 150              // Máximo de iterações
#define SKIP_HEADER     true             // Pular primeira linha (header)
// ─────────────────────────────────────

// Função para carregar CSV no host (retorna vector[N][D])
vector<vector<double>> load_csv(const string& filename) {
    ifstream in(filename);
    if (!in) {
        cerr << "Erro ao abrir arquivo: " << filename << "\n";
        exit(1);
    }
    vector<vector<double>> data;
    string line;
#if SKIP_HEADER
    getline(in, line);  // descarta header
#endif
    while (getline(in, line)) {
        while (!line.empty() && (line.back()=='\r' || line.back()=='\n' || line.back()==','))
            line.pop_back();
        if (line.empty()) continue;
        stringstream ss(line);
        vector<double> row;
        string cell;
        while (getline(ss, cell, ',')) {
            try { row.push_back(stod(cell)); } catch (...) {}
        }
        if (!row.empty()) row.pop_back();
        if (!row.empty()) data.emplace_back(move(row));
    }
    return data;
}

// Kernel CUDA: atribui cada ponto ao cluster mais próximo
__global__ void assign_labels(const double* data,
                              const double* centroids,
                              int* labels,
                              int N, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const double* p = data + idx * D;
    double best_dist = 1e300;
    int best_k = 0;
    for (int k = 0; k < K; ++k) {
        const double* c = centroids + k * D;
        double dist = 0;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            double diff = p[d] - c[d];
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_k = k; }
    }
    labels[idx] = best_k;
}

int main(int argc, char* argv[]) {
    int K = (argc >= 2 ? stoi(argv[1]) : DEFAULT_K);
    int max_iter = (argc >= 3 ? stoi(argv[2]) : DEFAULT_MAX_IT);

    // Carrega dados no host
    auto host_data = load_csv(DATA_FILE);
    int N = host_data.size();
    int D = host_data[0].size();
    cout << "→ Carreguei " << N << " amostras (dim=" << D << ")\n";

    // Flatten dados em vetor contínuo
    vector<double> flat_data(N * D);
    for (int i = 0; i < N; ++i) {
        memcpy(flat_data.data() + i * D, host_data[i].data(), D * sizeof(double));
    }

    // Host: centroids e labels
    vector<double> h_centroids(K * D);
    vector<int> h_labels(N, -1);

    // Inicialização randômica dos centróides
    mt19937_64 rng(1234);
    uniform_int_distribution<int> pick(0, N - 1);
    unordered_set<int> used;
    for (int k = 0; k < K; ) {
        int idx = pick(rng);
        if (used.insert(idx).second) {
            copy_n(flat_data.data() + idx * D, D, h_centroids.begin() + k * D);
            ++k;
        }
    }

    // Alocação de memória na GPU
    double *d_data, *d_centroids;
    int *d_labels;
    cudaMalloc(&d_data, N * D * sizeof(double));
    cudaMalloc(&d_centroids, K * D * sizeof(double));
    cudaMalloc(&d_labels, N * sizeof(int));

    // Cópia inicial de dados para GPU
    cudaMemcpy(d_data, flat_data.data(), N * D * sizeof(double), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Loop principal K-means
    for (int it = 0; it < max_iter; ++it) {
        cudaMemcpy(d_centroids, h_centroids.data(), K * D * sizeof(double), cudaMemcpyHostToDevice);
        assign_labels<<<blocks, threadsPerBlock>>>(d_data, d_centroids, d_labels, N, D, K);
        cudaDeviceSynchronize();
        cudaMemcpy(h_labels.data(), d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        // Recalcula centróides no host
        vector<vector<double>> sum(K, vector<double>(D, 0.0));
        vector<int> count(K, 0);
        bool changed = false;
        for (int i = 0; i < N; ++i) {
            int lbl = h_labels[i];
            ++count[lbl];
            for (int d = 0; d < D; ++d) {
                sum[lbl][d] += flat_data[i * D + d];
            }
        }
        for (int k = 0; k < K; ++k) {
            if (count[k] > 0) {
                for (int d = 0; d < D; ++d) {
                    double new_val = sum[k][d] / count[k];
                    if (fabs(new_val - h_centroids[k * D + d]) > 1e-6) changed = true;
                    h_centroids[k * D + d] = new_val;
                }
            }
        }
        if (!changed) {
            cout << "Convergiu em " << it << " iterações.\n";
            break;
        }
    }

    // Saída dos centróides
    cout << fixed << setprecision(4);
    for (int k = 0; k < K; ++k) {
        cout << "Centróide " << k << ": ";
        for (int d = 0; d < D; ++d) cout << h_centroids[k * D + d] << " ";
        cout << "\n";
    }

    // Impressão dos tamanhos de cada cluster
    vector<int> final_count(K, 0);
    for (int lbl : h_labels) ++final_count[lbl];
    for (int k = 0; k < K; ++k) {
        cout << "Cluster " << k << " tem " << final_count[k] << " pontos\n";
    }

    // Liberação de memória GPU
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    return 0;
}
