// open_mp_gpu.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

// -----------------------------------------------------------------------------
// Projeto: Implementação paralela de K-Means usando OpenMP Offload (GPU)
// Descrição: Versão para GPU usando OpenMP target.
// Carrega um CSV numérico, ignora opcionalmente o cabeçalho, e executa o K-Means
// com a etapa de atribuição (labels) offloadada para a GPU.
// Permite configurar clusters (K), iterações, arquivo de dados e threads.
// -----------------------------------------------------------------------------

// ─────────── CONFIGURAÇÃO ───────────────────────────────────────────────────────
// DATA_FILE:      Caminho padrão para o CSV de entrada
// DEFAULT_K:      Valor default de K (número de clusters)
// DEFAULT_MAX_IT: Valor default de iterações máximas
// SKIP_HEADER:    Ignorar a primeira linha (header) se true
// NUM_THREADS:    Threads CPU para as seções no host
// THREADS_GPU:    Número de threads por equipe na GPU (thread_limit)
// ────────────────────────────────────────────────────────────────────────────────
#define DATA_FILE       "covtype.csv"
#define DEFAULT_K       10
#define DEFAULT_MAX_IT  150
#define SKIP_HEADER     true
#define NUM_THREADS     32    // Ajuste o número de threads no host
#define THREADS_GPU     256   // Threads por equipe na GPU

// -----------------------------------------------------------------------------
// load_csv: lê um CSV numérico e retorna um vetor de amostras.
// -----------------------------------------------------------------------------
vector<vector<double>> load_csv(const string& filename) {
    ifstream in(filename);
    if (!in) {
        cerr << "Erro ao abrir arquivo: " << filename << endl;
        exit(1);
    }
    vector<vector<double>> data;
    string line;
#if SKIP_HEADER
    if (!getline(in, line)) {
        cerr << "Arquivo vazio ou sem header para pular" << endl;
        exit(1);
    }
#endif
    while (getline(in, line)) {
        while (!line.empty() && (line.back()=='\r' || line.back()=='\n' || line.back()==',')) 
            line.pop_back();
        if (line.empty()) continue;
        stringstream ss(line);
        vector<double> row;
        string cell;
        while (getline(ss, cell, ',')) {
            // trim
            cell.erase(cell.begin(),
                       find_if(cell.begin(), cell.end(),
                               [](unsigned char c){ return !isspace(c); }));
            cell.erase(find_if(cell.rbegin(), cell.rend(),
                               [](unsigned char c){ return !isspace(c); }).base(),
                       cell.end());
            try {
                row.push_back(stod(cell));
            } catch (...) { }
        }
        if (!row.empty()) row.pop_back();  // remove label
        if (!row.empty()) data.emplace_back(move(row));
    }
    return data;
}

// -----------------------------------------------------------------------------
// euclid: distância Euclidiana entre dois vetores.
// -----------------------------------------------------------------------------
double euclid(const double* a, const double* b, int D) {
    double sum = 0.0;
    for (int i = 0; i < D; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// -----------------------------------------------------------------------------
// main: configura OpenMP, carrega dados, inicializa centróides,
// offloada atribuição para GPU e faz update de centróides no host.
// -----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Threads no host
    omp_set_num_threads(NUM_THREADS);
    cout << "Threads host: " << omp_get_max_threads() << endl;

    // Argumentos: K, max_iter, arquivo
    int K = (argc >= 2 ? stoi(argv[1]) : DEFAULT_K);
    int max_iter = (argc >= 3 ? stoi(argv[2]) : DEFAULT_MAX_IT);
    string filename = (argc >= 4 ? argv[3] : DATA_FILE);

    // Carrega dados em memória host
    auto data = load_csv(filename);
    int N = data.size();
    if (N == 0) {
        cerr << "Nenhuma amostra carregada de " << filename << endl;
        return 1;
    }
    int D = data[0].size();
    cout << "→ Carreguei " << N << " amostras (dim=" << D << ")" << endl;
    if (K <= 0 || K > N) {
        cerr << "K inválido: " << K << endl;
        return 1;
    }

    // Flatten dos dados para GPU
    vector<double> flat_data(N * D);
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++)
            flat_data[i*D + d] = data[i][d];

    // Inicializa centróides host
    vector<double> centroids_flat(K * D);
    mt19937_64 rng(1234);
    uniform_int_distribution<int> pick(0, N-1);
    unordered_set<int> used;
    for (int k = 0; k < K; ) {
        int idx = pick(rng);
        if (used.insert(idx).second) {
            for (int d = 0; d < D; d++)
                centroids_flat[k*D + d] = flat_data[idx*D + d];
            k++;
        }
    }

    vector<int> labels(N, -1);

    // Abre região de dados GPU
    #pragma omp target data \
        map(to: flat_data[0:N*D]) \
        map(tofrom: centroids_flat[0:K*D], labels[0:N])
    {
        for (int iter = 0; iter < max_iter; iter++) {
            bool changed = false;

            // 1) Etapa de atribuição: offload para GPU
            #pragma omp target teams distribute parallel for thread_limit(THREADS_GPU) schedule(static) reduction(||:changed)
            for (int i = 0; i < N; i++) {
                const double* xi = &flat_data[i*D];
                double best_dist = numeric_limits<double>::infinity();
                int best_k = 0;
                // percorre centróides
                for (int k = 0; k < K; k++) {
                    const double* ck = &centroids_flat[k*D];
                    double dist = euclid(xi, ck, D);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_k = k;
                    }
                }
                if (labels[i] != best_k) {
                    labels[i] = best_k;
                    changed = true;
                }
            }

            // Se convergiu, sai
            if (!changed) {
                cout << "Convergência em " << iter << " iterações." << endl;
                break;
            }

            // 2) Host: recalcula centróides (média)
            vector<double> sum(K * D, 0.0);
            vector<int> count(K, 0);
            // soma por thread-safe
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++) {
                int c = labels[i];
                #pragma omp atomic
                count[c]++;
                for (int d = 0; d < D; d++) {
                    #pragma omp atomic
                    sum[c*D + d] += flat_data[i*D + d];
                }
            }
            // atualiza centróides
            for (int k = 0; k < K; k++) {
                if (count[k] == 0) continue;
                for (int d = 0; d < D; d++)
                    centroids_flat[k*D + d] = sum[k*D + d] / count[k];
            }
        }  // fim iterações
    }  // fim target data

    // Impressão final
    cout << fixed << setprecision(4);
    for (int k = 0; k < K; k++) {
        cout << "Centróide " << k << ": ";
        for (int d = 0; d < D; d++)
            cout << centroids_flat[k*D + d] << " ";
        cout << endl;
    }
    vector<int> cluster_size(K, 0);
    for (int lab : labels) cluster_size[lab]++;
    for (int k = 0; k < K; k++)
        cout << "Cluster " << k << " tem " << cluster_size[k] << " pontos" << endl;

    return 0;
}
