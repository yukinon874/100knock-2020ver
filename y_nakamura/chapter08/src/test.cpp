#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <primitiv/primitiv.h>

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

    Parameter pw({4, 300}, I::XavierUniform());
    float lr = 0.01;
    O::SGD optimizer(lr);
    optimizer.add(pw);

    int seed = 1234;
    //random_device rd;
    mt19937 rng(seed);

    const unsigned nepoch = 100, batch_size = 1;
    unsigned dim = train_x[0].size();
    vector<float> losses;
    
    for(int epoch = 0; epoch < nepoch; epoch++) {
        vector<unsigned> train_ids(train_x.size());
        iota(begin(train_ids), end(train_ids), 0);
        float train_loss = 0.0;
        for (unsigned ofs = 0; ofs < train_x.size(); ofs += batch_size) {
            const vector<unsigned> batch_ids(
                begin(train_ids) + ofs,
                begin(train_ids) + min<unsigned>(ofs + batch_size, train_x.size())
            );
            const auto x_batch = make_batch(train_x, batch_ids);
            const auto y_batch = make_batch(train_y, batch_ids);
            g.clear();
            const Node x = F::input<Node>(Shape({dim}, batch_ids.size()), x_batch);
            const Node w = F::parameter<Node>(pw);
            const Node res = F::matmul(w, x);
            const Node loss = F::softmax_cross_entropy(res, y_batch, 0);
            Node avg_loss = F::batch::mean(loss);
            train_loss += avg_loss.to_float() * batch_ids.size();
            optimizer.reset_gradients();
            loss.backward();
            optimizer.update();
        }
        cout << "epoch" << epoch << ": ";
        cout << train_loss / train_x.size() << endl;
        losses.emplace_back(train_loss / train_x.size());
    }

    function<float(vector<vector<float>>, vector<vector<unsigned>>)> calc_acc = [&](vector<vector<float>> x, vector<vector<unsigned>> y) {
        float acc = 0.0;
        float total_loss = 0.0;
        unsigned match = 0;
        int batch_size = 256;
        vector<unsigned> ids(x.size());
        iota(begin(ids), end(ids), 0);
        for (unsigned ofs = 0; ofs < x.size(); ofs += batch_size) {
            const vector<unsigned> batch_ids(
                begin(ids) + ofs,
                begin(ids) + min<unsigned>(ofs + batch_size, x.size())
            );
            const auto x_batch = make_batch(x, batch_ids);
            g.clear();
            const Node x = F::input<Node>(Shape({dim}, batch_ids.size()), x_batch);
            const Node w = F::parameter<Node>(pw);
            const Node res = F::matmul(w, x);
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
        const float accuracy = 100.0 * match / x.size();
        return accuracy;
    };
    cout << "train accuracy: " << calc_acc(train_x, train_y) << endl;
    cout << "valid accuracy: " << calc_acc(valid_x, valid_y) << endl;
    cout << "test accuracy: " << calc_acc(test_x, test_y) << endl;
    return 0;
}
