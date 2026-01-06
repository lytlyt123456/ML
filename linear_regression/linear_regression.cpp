#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

class Dataset {
private:
    std::vector<std::vector<double>> x_train; // 训练集样本特征
    std::vector<std::vector<double>> x_test; // 测试集样本特征
    std::vector<double> y_train; // 训练集标签
    std::vector<double> y_test; // 测试集标签
    
    int num_train_samples; // 训练集样本数量
    int num_test_samples; // 测试集样本数量

    std::string read_file(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        std::ostringstream buf;
        buf << ifs.rdbuf();
        return buf.str();
    }

    // 解析数据集字符串
    void parse_dataset_string(const std::string& dataset, int num_train_samples, int feature_dim) {
        int ptr = 0;
        int i = 0;
        int str_len = dataset.length();
        while (ptr < str_len) {
            if (dataset[ptr] == '\n') {
                ++i;
                ++ptr;
            }
            else if (ptr == 0 || dataset[ptr - 1] == '\n' || dataset[ptr - 1] == '\r') {
                double label = dataset[ptr] - '0';
                int tmp_ptr = ptr + 1;
                while (dataset[tmp_ptr] >= '0' && dataset[tmp_ptr] <= '9') {
                    label = label * 10.0 + dataset[tmp_ptr] - '0';
                    ++tmp_ptr;
                }
                if (dataset[tmp_ptr] == '.') {
                    double multiply = 0.1;
                    ++tmp_ptr;
                    while (dataset[tmp_ptr] >= '0' && dataset[tmp_ptr] <= '9') {
                        label += multiply * (dataset[tmp_ptr] - '0');
                        multiply *= 0.1;
                        ++tmp_ptr;
                    }
                }
                if (i < num_train_samples)
                    y_train[i] = label;
                else y_test[i - num_train_samples] = label;
                ptr = tmp_ptr;
            }
            else if (dataset[ptr] == ':') {
                double current_feature = 0.0; int current_feature_dim = 0;
                int tmp_ptr = ptr - 1;
                int multiply_dim = 1;
                while (dataset[tmp_ptr] >= '0' && dataset[tmp_ptr] <= '9') {
                    current_feature_dim += multiply_dim * (dataset[tmp_ptr] - '0');
                    multiply_dim *= 10;
                    --tmp_ptr;
                }
                --current_feature_dim;

                double signal = 1.0;
                if (dataset[ptr + 1] == '-') {
                    ++ptr;
                    signal = -1.0;
                }

                tmp_ptr = ptr + 1;
                while (tmp_ptr < str_len && dataset[tmp_ptr] >= '0' && dataset[tmp_ptr] <= '9') {
                    current_feature = current_feature * 10.0 + dataset[tmp_ptr] - '0';
                    ++tmp_ptr;
                }
                if (tmp_ptr < str_len && dataset[tmp_ptr] == '.') {
                    double multiply = 0.1;
                    ++tmp_ptr;
                    while (tmp_ptr < str_len && dataset[tmp_ptr] >= '0' && dataset[tmp_ptr] <= '9') {
                        current_feature += multiply * (dataset[tmp_ptr] - '0');
                        multiply *= 0.1;
                        ++tmp_ptr;
                    }
                }

                if (i < num_train_samples)
                    x_train[i][current_feature_dim] = signal * current_feature;
                else x_test[i - num_train_samples][current_feature_dim] = signal * current_feature;
                ptr = tmp_ptr;
            }
            else ++ptr;
        }
    }

public:
    Dataset(const std::string& dataset_path, double train_set_percent, int num_samples, int feature_dim) {
        int num_train_samples = train_set_percent * num_samples;
        int num_test_samples = num_samples - num_train_samples;
        this->num_train_samples = num_train_samples;
        this->num_test_samples = num_test_samples;

        x_train = std::vector<std::vector<double>>(num_train_samples, std::vector<double>(feature_dim + 1));
        x_test = std::vector<std::vector<double>>(num_test_samples, std::vector<double>(feature_dim + 1));
        y_train = std::vector<double>(num_train_samples);
        y_test = std::vector<double>(num_test_samples);

        std::string dataset = read_file(dataset_path);

        // 将样本特征的最后一个维度的值设为1.0（与偏置b相乘）
        for (int i = 0; i < num_train_samples; ++i)
            x_train[i][feature_dim] = 1.0;
        for (int i = 0; i < num_test_samples; ++i)
            x_test[i][feature_dim] = 1.0;

        // 解析数据集字符串
        parse_dataset_string(dataset, num_train_samples, feature_dim);
    }

    double get_x_train(int i, int j) {
        return x_train[i][j];
    }

    double get_x_test(int i, int j) {
        return x_test[i][j];
    }

    double get_y_train(int i) {
        return y_train[i];
    }

    double get_y_test(int i) {
        return y_test[i];
    }

    std::vector<std::vector<double>> get_x_train() { return x_train; }

    std::vector<std::vector<double>> get_x_test() { return x_test; }

    std::vector<double> get_y_train() { return y_train; }

    std::vector<double> get_y_test() { return y_test; }

    int get_num_train_samples() { return num_train_samples; }

    int get_num_test_samples() { return num_test_samples;  }
};

class HousingPricePredictor {
private:
    std::vector<double> param; // 模型参数，feature_dim + 1维向量
    int feature_dim; // 特征维度

    std::vector<std::vector<double>> matrix_transpose(
        const std::vector<std::vector<double>>& mat
    ) {
        int m = mat.size();
        int n = mat[0].size();

        std::vector<std::vector<double>> res(n, std::vector<double>(m));

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                res[j][i] = mat[i][j];

        return res;
    }

    std::vector<std::vector<double>> matrix_multiply(
        const std::vector<std::vector<double>>& m_x,
        const std::vector<std::vector<double>>& m_y
    ) {
        int m = m_x.size();
        int n = m_x[0].size();
        int s = m_y[0].size();

        std::vector<std::vector<double>> res(m, std::vector<double>(s));

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < s; ++j) {
                res[i][j] = 0;
                for (int k = 0; k < n; ++k)
                    res[i][j] += m_x[i][k] * m_y[k][j];
            }
        }

        return res;
    }

    std::vector<std::vector<double>> matrix_plus(
        const std::vector<std::vector<double>>& m_x,
        const std::vector<std::vector<double>>& m_y
    ) {
        int m = m_x.size();
        int n = m_x[0].size();

        std::vector<std::vector<double>> res(m, std::vector<double>(n));

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                res[i][j] = m_x[i][j] + m_y[i][j];

        return res;
    }

    std::vector<std::vector<double>> matrix_subtract(
        const std::vector<std::vector<double>>& m_x,
        const std::vector<std::vector<double>>& m_y
    ) {
        int m = m_x.size();
        int n = m_x[0].size();

        std::vector<std::vector<double>> res(m, std::vector<double>(n));

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                res[i][j] = m_x[i][j] - m_y[i][j];

        return res;
    }

    // 求矩阵的逆
    std::vector<std::vector<double>> matrix_inverse(
        const std::vector<std::vector<double>>& mat
    ) {
        int n = mat.size();

        // 增广矩阵
        std::vector<std::vector<double>> tmp_mat(n, std::vector<double>(n + n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                tmp_mat[i][j] = mat[i][j];
        for (int i = 0; i < n; ++i)
            tmp_mat[i][n + i] = 1.0;

        // 高斯消元变成上三角矩阵
        for (int i = 0; i <= n - 2; ++i) {
            double max_value = std::abs(tmp_mat[i][i]);
            int max_value_row = i;
            for (int j = i + 1; j <= n - 1; ++j) {
                if (std::abs(tmp_mat[j][i]) > max_value) {
                    max_value = std::abs(tmp_mat[j][i]);
                    max_value_row = j;
                }
            }
            if (i != max_value_row)
                for (int j = i; j <= n + n - 1; ++j)
                    std::swap(tmp_mat[i][j], tmp_mat[max_value_row][j]);
            for (int j = i + 1; j <= n - 1; ++j) {
                double multiply = tmp_mat[j][i] / tmp_mat[i][i];
                for (int k = i; k <= n + n - 1; ++k)
                    tmp_mat[j][k] = tmp_mat[j][k] - multiply * tmp_mat[i][k];
            }
        }

        // 使主对角线上元素的值均为1
        for (int i = 0; i <= n - 1; ++i) {
            double divider = tmp_mat[i][i];
            for (int j = i; j <= n + n - 1; ++j)
                tmp_mat[i][j] = tmp_mat[i][j] / divider;
        }

        // 变成单位矩阵
        for (int i = n - 1; i >= 1; --i) {
            for (int j = i - 1; j >= 0; --j) {
                double multiply = tmp_mat[j][i];
                for (int k = i; k <= n + n - 1; ++k)
                    tmp_mat[j][k] = tmp_mat[j][k] - multiply * tmp_mat[i][k];
            }
        }

        // 增广矩阵右半部为原矩阵的逆
        std::vector<std::vector<double>> res(n, std::vector<double>(n));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                res[i][j] = tmp_mat[i][n + j];

        return res;
    }

    // 计算梯度
    std::vector<double> compute_grad(const std::vector<std::vector<double>>& x,
        const std::vector<double>& y) {
        std::vector<std::vector<double>> param_2d;
        param_2d.push_back(param);
        std::vector<std::vector<double>> y_2d;
        y_2d.push_back(y);

        std::vector<std::vector<double>> result = 
            matrix_subtract(matrix_multiply(matrix_multiply(param_2d, matrix_transpose(x)), x), 
            matrix_multiply(y_2d, x));

        return result[0];
    }

public:
    HousingPricePredictor(int feature_dim) {
        this->feature_dim = feature_dim;
        param = std::vector<double>(feature_dim + 1);
    }

    // 闭式解法推导参数的值
    void train_closed_form(const std::vector<std::vector<double>>& x_train, 
        const std::vector<double>& y_train, double lambda) {
        std::vector<std::vector<double>> x_t_mul_x
            = matrix_multiply(matrix_transpose(x_train), x_train);

        for (int i = 0; i < x_t_mul_x.size(); ++i)
            x_t_mul_x[i][i] += lambda;

        std::vector<std::vector<double>> x_t_mul_x_inverse = matrix_inverse(x_t_mul_x);

        std::vector<std::vector<double>>tmptmp = matrix_multiply(x_t_mul_x, x_t_mul_x_inverse);
        int nnn = x_t_mul_x.size();

        std::vector<std::vector<double>> x_t_mul_x_inverse_mul_x_t
            = matrix_multiply(x_t_mul_x_inverse, matrix_transpose(x_train));

        std::vector<std::vector<double>> y_train_2d;
        y_train_2d.push_back(y_train);
        std::vector<std::vector<double>> y_t = matrix_transpose(y_train_2d);

        std::vector<std::vector<double>> res = matrix_multiply(x_t_mul_x_inverse_mul_x_t, y_t);
        for (int i = 0; i < feature_dim + 1; ++i)
            param[i] = res[i][0];

        int n = x_train.size();
        std::vector<double> y_pred(n);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < feature_dim + 1; ++j)
                y_pred[i] += x_train[i][j] * param[j];

        std::cout << "训练集上的预测结果与标签对比：\n\n";
        for (int i = 0; i < n; ++i)
            std::cout << i << "、" << "预测结果：" << y_pred[i] << "，标签：" << y_train[i] << "\n";
        double loss = 0;
        for (int i = 0; i < n; ++i)
            loss += (y_pred[i] - y_train[i])* (y_pred[i] - y_train[i]);
        loss /= n;
        std::cout << "\n\n训练集上的损失函数：" << loss << "\n";
    }

    // 梯度下降法训练
    std::vector<std::vector<double>>  train_gd(
        const std::vector<std::vector<double>>& x_train,
        const std::vector<double>& y_train,
        const std::vector<std::vector<double>>& x_test,
        const std::vector<double>& y_test,
        double learning_rate, double epochs
    ) {
        std::vector<double> loss_trains;
        std::vector<double> loss_tests;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<double> grad = compute_grad(x_train, y_train);
            for (int i = 0; i < feature_dim + 1; ++i)
                param[i] -= learning_rate * grad[i];

            int n_train = x_train.size();
            std::vector<double> y_pred_train(n_train);

            for (int i = 0; i < n_train; ++i)
                for (int j = 0; j < feature_dim + 1; ++j)
                    y_pred_train[i] += x_train[i][j] * param[j];

            double loss_train = 0;
            for (int i = 0; i < n_train; ++i)
                loss_train += (y_pred_train[i] - y_train[i]) * (y_pred_train[i] - y_train[i]);
            loss_train /= n_train;
            std::cout << "[epoch " << epoch + 1 << "] loss_train: " << loss_train << ", ";
            loss_trains.push_back(loss_train);

            int n_test = x_test.size();
            std::vector<double> y_pred_test(n_test);

            for (int i = 0; i < n_test; ++i)
                for (int j = 0; j < feature_dim + 1; ++j)
                    y_pred_test[i] += x_test[i][j] * param[j];

            double loss_test = 0;
            for (int i = 0; i < n_test; ++i)
                loss_test += (y_pred_test[i] - y_test[i]) * (y_pred_test[i] - y_test[i]);
            loss_test /= n_test;
            std::cout << "loss_test: " << loss_test << "\n";
            loss_tests.push_back(loss_test);
        }

        std::vector<std::vector<double>> res;
        res.push_back(loss_trains);
        res.push_back(loss_tests);

        return res;
    }

    void reset_param() {
        for (int i = 0; i < feature_dim + 1; ++i)
            param[i] = 0.0;
    }

    void test(const std::vector<std::vector<double>>& x_test,
        const std::vector<double>& y_test) {
        int n = x_test.size();

        std::vector<double> y_pred(n);

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < feature_dim + 1; ++j)
                y_pred[i] += x_test[i][j] * param[j];

        std::cout << "\n\n\n测试集上的预测结果与标签对比：\n\n";
        for (int i = 0; i < n; ++i)
            std::cout << i << "、" << "预测结果：" << y_pred[i] << "，标签：" << y_test[i] << "\n";
        double loss = 0;
        for (int i = 0; i < n; ++i)
            loss += (y_pred[i] - y_test[i]) * (y_pred[i] - y_test[i]);
        loss /= n;
        std::cout << "\n\n测试集上的损失函数：" << loss << "\n";
    }
};

void plot_data(const std::vector<double>& data, const std::string& loss_type) {
    // 创建数据文件
    std::ofstream dataFile("data_" + loss_type + ".txt");

    for (int i = 0; i < data.size(); i++)
        dataFile << i << " " << data[i] << std::endl;
    dataFile.close();

    // 创建GNUplot脚本
    std::ofstream scriptFile("plot_script_" + loss_type + ".plt");
    scriptFile << "set title '" << (loss_type == "loss_train" ? "loss train" : "loss test") << " 变化曲线图'" << std::endl;
    scriptFile << "set xlabel 'epoch'" << std::endl;
    scriptFile << "set ylabel '" << (loss_type == "loss_train" ? "loss train" : "loss test") << "'" << std::endl;
    scriptFile << "set grid" << std::endl;
    scriptFile << "plot 'data_" << loss_type << ".txt' with linespoints title '"
        << (loss_type == "loss_train" ? "loss train" : "loss test") << " 变化'" << std::endl;
    scriptFile << "pause -1" << std::endl; // 保持图形显示
    scriptFile.close();

    // 执行GNUplot
    if (loss_type == "loss_train")
        system("gnuplot plot_script_loss_train.plt");
    else system("gnuplot plot_script_loss_test.plt");
}

int main() {
    Dataset dataset("dataset.txt", 0.8, 506, 13);
    HousingPricePredictor model(13);

    // 闭式解法
    std::cout << "闭式解法：\n\n";
    model.train_closed_form(dataset.get_x_train(), dataset.get_y_train(), 1e-5);
    model.test(dataset.get_x_test(), dataset.get_y_test());

    model.reset_param();

    // 梯度下降法
    std::cout << "\n\n\n梯度下降法：\n\n";
    std::vector<std::vector<double>> loss = model.train_gd(
        dataset.get_x_train(), dataset.get_y_train(),
        dataset.get_x_test(), dataset.get_y_test(), 1e-5, 2000
    );

    plot_data(loss[0], "loss_train");
    plot_data(loss[1], "loss_test");

    return 0;
}