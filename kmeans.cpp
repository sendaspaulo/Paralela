#include <bits/stdc++.h>
using namespace std;

// ─────────── CONFIGURAÇÃO ───────────
// Defina aqui o CSV, o valor default de K, max de iterações e se há header
#define DATA_FILE       "covtype.csv"
#define DEFAULT_K       10
#define DEFAULT_MAX_IT  150
#define SKIP_HEADER     true
// ─────────────────────────────────────

// Lê CSV numérico, opcionalmente pula header e remove a última coluna (rótulo)
vector<vector<double>> load_csv(const string& filename) {
    ifstream in(filename);
    if (!in) {
        cerr << "Erro ao abrir arquivo: " << filename << endl;
        exit(1);
    }
    vector<vector<double>> data;
    string line;
#if SKIP_HEADER
    // pula a primeira linha de cabeçalho
    if (!getline(in, line)) {
        cerr << "Arquivo vazio ou sem header para pular\n";
        exit(1);
    }
#endif
    while (getline(in, line)) {
        // limpa CR/LF e vírgula final
        while (!line.empty() && (line.back()=='\r' || line.back()=='\n' || line.back()==','))
            line.pop_back();
        if (line.empty()) continue;

        stringstream ss(line);
        vector<double> row;
        string cell;
        while (getline(ss, cell, ',')) {
            // trim espaços
            cell.erase(cell.begin(), find_if(cell.begin(), cell.end(), [](unsigned char c){ return !isspace(c); }));
            cell.erase(find_if(cell.rbegin(), cell.rend(), [](unsigned char c){ return !isspace(c); }).base(), cell.end());
            try {
                row.push_back(stod(cell));
            } catch (invalid_argument&) {
                // ignora qualquer token não convertido
            }
        }
        // remove a última coluna (suposto rótulo)
        if (!row.empty()) row.pop_back();
        if (!row.empty()) data.emplace_back(move(row));
    }
    return data;
}

// Distância Euclidiana
double euclid(const vector<double>& a, const vector<double>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = a[i] - b[i];
        sum += d*d;
    }
    return sqrt(sum);
}

int main(int argc, char* argv[]) {
    // argumentos: [K] [max_iter] [arquivo]
    int K        = (argc >= 2 ? stoi(argv[1]) : DEFAULT_K);
    int max_iter = (argc >= 3 ? stoi(argv[2]) : DEFAULT_MAX_IT);
    string filename = (argc >= 4 ? argv[3] : DATA_FILE);

    auto data = load_csv(filename);
    int N = data.size();
    if (N == 0) {
        cerr << "Nenhuma amostra carregada de " << filename << "\n";
        return 1;
    }
    int D = data[0].size();
    cout << "→ Carreguei " << N << " amostras de " << filename
         << " (dim=" << D << ")\n";

    if (K <= 0 || K > N) {
        cerr << "Valor de K inválido: " << K << "\n";
        return 1;
    }

    // inicializa centróides distintos
    vector<vector<double>> centroids;
    mt19937_64 rng(1234);
    uniform_int_distribution<int> pick(0, N-1);
    unordered_set<int> used;
    while ((int)centroids.size() < K) {
        int idx = pick(rng);
        if (used.insert(idx).second)
            centroids.push_back(data[idx]);
    }

    vector<int> labels(N, -1);
    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;
        // atribuição
        for (int i = 0; i < N; i++) {
            double best = numeric_limits<double>::infinity();
            int who = 0;
            for (int k = 0; k < K; k++) {
                double d = euclid(data[i], centroids[k]);
                if (d < best) { best = d; who = k; }
            }
            if (labels[i] != who) { labels[i] = who; changed = true; }
        }
        if (!changed) {
            cout << "Convergiu em " << iter << " iterações.\n";
            break;
        }
        // recomputa centróides
        vector<vector<double>> sum(K, vector<double>(D, 0.0));
        vector<int> count(K, 0);
        for (int i = 0; i < N; i++) {
            int k = labels[i];
            count[k]++;
            for (int d = 0; d < D; d++) sum[k][d] += data[i][d];
        }
        for (int k = 0; k < K; k++) {
            if (count[k] == 0) continue;
            for (int d = 0; d < D; d++)
                centroids[k][d] = sum[k][d] / count[k];
        }
    }

    // saída
    cout << fixed << setprecision(4);
    for (int k = 0; k < K; k++) {
        cout << "Centróide " << k << ": ";
        for (double v : centroids[k]) cout << v << " ";
        cout << "\n";
    }
    vector<int> sz(K, 0);
    for (int l : labels) sz[l]++;
    for (int k = 0; k < K; k++)
        cout << "Cluster " << k << " tem " << sz[k] << " pontos\n";

    return 0;
}