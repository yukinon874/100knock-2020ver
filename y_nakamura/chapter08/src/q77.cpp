#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <primitiv/primitiv.h>
#include <primitiv/primitiv_cuda.h>
#include <functional>
#include <time.h>

using namespace std;
using namespace primitiv;
namespace F = primitiv::functions;
namespace I = primitiv::initializers;
namespace O = primitiv::optimizers;


template<typename T>
vector<vector<T>> load_data(string path) {
    fstream ifs(path);
    stringstream ss;
    string line;
    getline(ifs, line);
    ss << line;
    int size, dim;
    ss >> size >> dim;
    vector<vector<T>> data(size, vector<T> (dim));
    for (int iter = 0; iter < size; iter++) {
        ss.clear(stringstream::goodbit);
        getline(ifs, line);
        assert(line != "");
        ss << line;
        for (int i = 0; i < dim; i++) {
            ss >> data[iter][i];
        }
    }
    return data;
}

template<typename T>
vector<T> make_batch(const vector<vector<T>> &x, const vector<unsigned> &batch_ids) {
    vector<T> data;
    for(int i = 0; i < batch_ids.size(); i++) {
        for(int j = 0; j < x[batch_ids[i]].size(); j++) {
            data.emplace_back(x[batch_ids[i]][j]);
        }
    }
    return data;
};


int main(int argc, char *argv[]) {
    vector<vector<float>> train_x = load_data<float>("./work/train_x.data");
    vector<vector<unsigned>> train_y = load_data<unsigned>("./work/train_y.data");
    vector<vector<float>> valid_x = load_data<float>("./work/valid_x.data");
    vector<vector<unsigned>> valid_y = load_data<unsigned>("./work/valid_y.data");
    vector<vector<float>> test_x = load_data<float>("./work/test_x.data");
    vector<vector<unsigned>> test_y = load_data<unsigned>("./work/test_y.data");

    devices::Naive dev;
    Device::set_default(dev);

    Graph g;
    Graph::set_default(g);

    for(int batch_size=1; batch_size <= 1024; batch_size *= 2){
        Parameter pw({4, 300}, I::XavierUniform());
        float lr = 0.01;
        O::SGD optimizer(lr);
        optimizer.add(pw);

        random_device rd;
        mt19937 rng(rd());

        const unsigned nepoch = 100;
        unsigned dim = train_x[0].size();

        function<pair<float, float>(vector<vector<float>>, vector<vector<unsigned>>, bool)> execute = \
            [&](vector<vector<float>> x, vector<vector<unsigned>> y, bool is_train) {

            vector<unsigned> data_ids(x.size());
            iota(begin(data_ids), end(data_ids), 0);
            shuffle(begin(data_ids), end(data_ids), rng);
            unsigned match = 0;
            float total_loss = 0.0;
            for (unsigned ofset = 0; ofset < x.size(); ofset += batch_size) {
                const vector<unsigned> batch_ids(
                        begin(data_ids) + ofset,
                        begin(data_ids) + min<unsigned>(ofset + batch_size, x.size())
                    );
                const auto x_batch = make_batch(x, batch_ids);
                const auto y_batch = make_batch(y, batch_ids);
                g.clear();
                const Node x = F::input<Node>(Shape({dim}, batch_ids.size()), x_batch);
                const Node w = F::parameter<Node>(pw);
                const Node res = F::matmul(w, x);
                const Node loss = F::softmax_cross_entropy(res, y_batch, 0);
                Node avg_loss = F::batch::mean(loss);
                total_loss += avg_loss.to_float() * batch_ids.size();
                optimizer.reset_gradients();
                if(is_train) {
                    loss.backward();
                    optimizer.update();
                }
                vector<float> y_val = res.to_vector();
                for (unsigned i = 0; i < batch_ids.size(); i++) {
                    float maxval = -1e10;
                    int argmax = -1;
                    for (unsigned j = 0; j < 4; j++) {
                        float v = y_val[j + i * 4];
                        if (v > maxval) maxval = v, argmax = static_cast<int>(j);
                    }
                    if (argmax == y[batch_ids[i]][0]) match++;
                }
            }
            float loss = total_loss / x.size();
            float accuracy = 100.0 * match / x.size();
            return make_pair(loss, accuracy);
        };


        clock_t train_start = clock();
        for(int epoch = 0; epoch < nepoch; epoch++) {
            clock_t start = clock();
            pair<float, float> train_res = execute(train_x, train_y, true);

            pair<float, float> valid_res = execute(valid_x, valid_y, false);
        }
        float train_time = static_cast<float> (clock() - train_start) / CLOCKS_PER_SEC;
        cout << "batch_size\t" << batch_size << "\ttrain_time\t" << train_time << " [sec]" << endl;
    }
    return 0;
}
