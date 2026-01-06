#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

class Dataset {
private:
    std::vector<std::vector<double>> x_train; // 训练集样本特征
    std::vector<std::vector<double>> x_test; // 测试集样本特征
    std::vector<int> y_train; // 训练集标签
    std::vector<int> y_test; // 测试集标签

    int num_train_samples; // 训练集样本数量
    int num_test_samples; // 测试集样本数量

    bool is_two_classes;

    std::string read_file(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        std::ostringstream buf;
        buf << ifs.rdbuf();
        return buf.str();
    }

    // 解析数据集字符串
    void parse_dataset_string(const std::string& dataset, int num_train_samples) {
        int ptr = 0;
        int i = 0;
        int str_len = dataset.length();
        while (ptr < str_len) {
            if (dataset[ptr] == '\n') {
                ++i;
                ++ptr;
            }
            else if (ptr == 0 || dataset[ptr - 1] == '\n') {
                int label = dataset[ptr] - '0';
                if (is_two_classes) {
                    if (label == 2) label = 0;
                    else label = 1;
                }
                else --label;
                if (i < num_train_samples)
                    y_train[i] = label;
                else y_test[i - num_train_samples] = label;
                ++ptr;
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
    Dataset(const std::string& dataset_path, double train_set_percent, int num_samples, int feature_dim,
        bool is_two_classes) {
        int num_train_samples = train_set_percent * num_samples;
        int num_test_samples = num_samples - num_train_samples;
        this->num_train_samples = num_train_samples;
        this->num_test_samples = num_test_samples;
        this->is_two_classes = is_two_classes;

        x_train = std::vector<std::vector<double>>(num_train_samples, std::vector<double>(feature_dim + 1));
        x_test = std::vector<std::vector<double>>(num_test_samples, std::vector<double>(feature_dim + 1));
        y_train = std::vector<int>(num_train_samples);
        y_test = std::vector<int>(num_test_samples);

        std::string dataset = read_file(dataset_path);

        // 将样本特征的最后一个维度的值设为1.0（与偏置b相乘）
        for (int i = 0; i < num_train_samples; ++i)
            x_train[i][feature_dim] = 1.0;
        for (int i = 0; i < num_test_samples; ++i)
            x_test[i][feature_dim] = 1.0;

        // 解析数据集字符串
        parse_dataset_string(dataset, num_train_samples);
    }

    double get_x_train(int i, int j) { return x_train[i][j]; }

    double get_x_test(int i, int j) { return x_test[i][j]; }

    int get_y_train(int i) { return y_train[i]; }

    int get_y_test(int i) { return y_test[i]; }

    std::vector<std::vector<double>> get_x_train() { return x_train; }

    std::vector<std::vector<double>> get_x_test() { return x_test; }

    std::vector<int> get_y_train() { return y_train; }

    std::vector<int> get_y_test() { return y_test; }

    int get_num_train_samples() { return num_train_samples; }

    int get_num_test_samples() { return num_test_samples; }
};

class SupportVectorMachine {
private:
    std::vector<double> weights; // 权重
    double bias; // 偏置

    int feature_dim; // 特征维度

    std::vector<double> compute_grad(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train, double penalty_term) {

        std::vector<double> grad(feature_dim + 1);
        // 前feature_dim个元素是对weights的梯度，最后一个元素是对bias的梯度

        int n = x_train.size();
        for (int i = 0; i < n; ++i) {
            double tmp = 0;
            for (int j = 0; j < feature_dim; ++j)
                tmp += weights[j] * x_train[i][j];
            tmp = 1 - (y_train[i] == 1 ? 1 : -1) * (tmp + bias);
            if (tmp >= 0) {
                for (int j = 0; j < feature_dim; ++j)
                    grad[j] += -(y_train[i] == 1 ? 1 : -1) * x_train[i][j];
                grad[feature_dim] += -(y_train[i] == 1 ? 1 : -1);
            }
        }
        for (int i = 0; i < feature_dim; ++i)
            grad[i] = weights[i] + penalty_term / n * grad[i];
        grad[feature_dim] *= penalty_term / n;

        return grad;
    }

public:
    SupportVectorMachine(int feature_dim) {
        weights = std::vector<double>(feature_dim);
        bias = 0.0;
        this->feature_dim = feature_dim;
    }

    void train(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train,
        const std::vector<std::vector<double>>& x_test,
        const std::vector<int>& y_test,
        int num_epochs, double learning_rate, double penalty_term) {

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::vector<double> grad = compute_grad(x_train, y_train, penalty_term);
            for (int i = 0; i < feature_dim; ++i)
                weights[i] -= learning_rate * grad[i];
            bias -= learning_rate * grad[feature_dim];

            // 计算训练集和测试集的预测准确率
            int n_train = x_train.size();
            int cnt_correct_train = 0;
            for (int i = 0; i < n_train; ++i) {
                double y_pred = 0.0;
                for (int j = 0; j < feature_dim; ++j)
                    y_pred += x_train[i][j] * weights[j];
                y_pred += bias;
                if (y_pred > 0 && y_train[i] == 1 || y_pred < 0 && y_train[i] == 0)
                    ++cnt_correct_train;
            }
            double acc_train = 1.0 * cnt_correct_train / n_train;

            int n_test = x_test.size();
            int cnt_correct_test = 0;
            for (int i = 0; i < n_test; ++i) {
                double y_pred = 0.0;
                for (int j = 0; j < feature_dim; ++j)
                    y_pred += x_test[i][j] * weights[j];
                y_pred += bias;
                if (y_pred > 0 && y_test[i] == 1 || y_pred < 0 && y_test[i] == 0)
                    ++cnt_correct_test;
            }
            double acc_test = 1.0 * cnt_correct_test / n_test;

            std::cout << "[epoch " << epoch + 1 << "] accuracy_train_set = "
                << acc_train << ", accuracy_test_set = " << acc_test << "\n";
        }

        std::vector<double> y_pred_test;
        int n_test = x_test.size();
        int cnt_correct_test = 0;
        for (int i = 0; i < n_test; ++i) {
            double y_pred = 0.0;
            for (int j = 0; j < feature_dim; ++j)
                y_pred += x_test[i][j] * weights[j];
            y_pred += bias;
            y_pred = 1.0 / (1.0 + std::exp(-y_pred));
            y_pred_test.push_back(y_pred);
        }

        // 绘制ROC曲线
        int pos_class_num = 0;
        int neg_class_num = 0;
        for (int i = 0; i < n_test; ++i) {
            if (y_test[i] == 1) ++pos_class_num;
            else ++neg_class_num;
        }

        struct RocDot {
            double false_positive_rate;
            double true_positive_rate;
        };

        std::vector<RocDot> roc;

        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.1) {
            int pos_pos_num = 0; // 正类样本被预测为正类的个数
            int neg_pos_num = 0; // 负类样本被预测为正类的个数
            for (int i = 0; i < n_test; ++i) {
                if (y_pred_test[i] >= threshold && y_test[i] == 1)
                    ++pos_pos_num;
                else if (y_pred_test[i] >= threshold && y_test[i] == 0)
                    ++neg_pos_num;
            }
            double fpr = 1.0 * neg_pos_num / neg_class_num;
            double tpr = 1.0 * pos_pos_num / pos_class_num;
            roc.push_back({ fpr, tpr });
        }

        std::cout << "\nROC:\n";
        for (auto roc_dot : roc)
            std::cout << roc_dot.false_positive_rate << " " << roc_dot.true_positive_rate << "\n";
    }

    void train_without_eval(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train,
        int num_epochs, double learning_rate, double penalty_term) {

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::vector<double> grad = compute_grad(x_train, y_train, penalty_term);
            for (int i = 0; i < feature_dim; ++i)
                weights[i] -= learning_rate * grad[i];
            bias -= learning_rate * grad[feature_dim];
        }
    }

    int eval(const std::vector<double>& x) {
        double y_pred = 0.0;
        for (int i = 0; i < feature_dim; ++i)
            y_pred += weights[i] * x[i];
        y_pred += bias;
        if (y_pred > 0)
            return 1;
        else if (y_pred < 0)
            return -1;
        else return 0;
    }
};

class SupportVectorMachine_multi_classes { // One-vs-One 方法实现多类分类
private:
    class Classifier {
    public:
        int positive_class; // 正类的编号
        int negative_class; // 负类的编号
        SupportVectorMachine* svm; // 用于二分类的支持向量机

        Classifier(int positive_class, int negative_class, int feature_dim) {
            this->positive_class = positive_class;
            this->negative_class = negative_class;
            svm = new SupportVectorMachine(feature_dim);
        }

        ~Classifier() { delete svm; }
    };

    std::vector<Classifier*> classifiers; // 存储所有分类器的数组
    int feature_dim; // 特征维度
    int class_num; // 类别个数
    int classifier_num; // 分类器的个数

public:
    SupportVectorMachine_multi_classes(int feature_dim, int class_num) {
        this->feature_dim = feature_dim;
        this->class_num = class_num;
        classifier_num = (class_num & 1) ? ((class_num - 1) / 2 * class_num) : (class_num / 2 * (class_num - 1));
        for (int i = 0; i <= class_num - 2; ++i)
            for (int j = i + 1; j <= class_num - 1; ++j) {
                Classifier* c = new Classifier(i, j, feature_dim);
                classifiers.push_back(c);
            }
    }

    ~SupportVectorMachine_multi_classes() {
        for (Classifier* classifier : classifiers)
            delete classifier;
    }

    void train(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train,
        const std::vector<std::vector<double>>& x_test,
        const std::vector<int>& y_test,
        int num_epochs, double learning_rate, double penalty_term) {

        for (auto classifier : classifiers) {
            int pos_c = classifier->positive_class;
            int neg_c = classifier->negative_class;
            SupportVectorMachine* svm = classifier->svm;

            std::vector<std::vector<double>> x_train_2cls;
            std::vector<int> y_train_2cls;

            for (int i = 0; i < x_train.size(); ++i) {
                if (y_train[i] == pos_c) {
                    x_train_2cls.push_back(x_train[i]);
                    y_train_2cls.push_back(1);
                }
                else if (y_train[i] == neg_c) {
                    x_train_2cls.push_back(x_train[i]);
                    y_train_2cls.push_back(0);
                }
            }

            svm->train_without_eval(x_train_2cls, y_train_2cls, num_epochs, learning_rate, penalty_term);
        }

        int n_train = x_train.size();
        int cnt_correct_train = 0;
        for (int i = 0; i < n_train; ++i)
            cnt_correct_train += eval(x_train[i], y_train[i]);
        double acc_train = 1.0 * cnt_correct_train / n_train;

        std::vector<std::vector<int>> confusion_matrix(class_num, std::vector<int>(class_num));

        int n_test = x_test.size();
        int cnt_correct_test = 0;
        for (int i = 0; i < n_test; ++i)
            cnt_correct_test += eval_test(x_test[i], y_test[i], confusion_matrix);
        double acc_test = 1.0 * cnt_correct_test / n_test;

        std::cout << "[epoch " << num_epochs << "] accuracy_train_set = "
            << acc_train << ", accuracy_test_set = " << acc_test << "\n";

        for (int i = 0; i < class_num; ++i) {
            for (int j = 0; j < class_num; ++j)
                std::cout << confusion_matrix[i][j] << " ";
            std::cout << "\n";
        }
    }

    int eval(const std::vector<double>& x, int y) {
        std::vector<int> count(class_num);

        for (auto classifier : classifiers) {
            int pos_c = classifier->positive_class; // 正类编号
            int neg_c = classifier->negative_class; // 负类编号
            SupportVectorMachine* svm = classifier->svm; // 支持向量机
            int pred = svm->eval(x); // pred = weights.transpose * x + bias
            if (pred > 0) ++count[pos_c]; // 预测值为正，则正类的票数加1
            else if (pred < 0) ++count[neg_c]; // 预测值为负，则负类的票数加1
        }

        // 选取票数最高的类别作为预测类别
        int max_value = -1; int max_pos = -1;
        for (int i = 0; i < class_num; ++i) {
            if (count[i] > max_value) {
                max_value = count[i];
                max_pos = i;
            }
        }

        // 如果预测类别与标签值相同，则预测成功，否则预测失败
        if (max_pos == y) return 1;
        else return 0;
    }

    int eval_test(const std::vector<double>& x, int y, std::vector<std::vector<int>>& confusion_matrix) {
        std::vector<int> count(class_num);

        for (auto classifier : classifiers) {
            int pos_c = classifier->positive_class; // 正类编号
            int neg_c = classifier->negative_class; // 负类编号
            SupportVectorMachine* svm = classifier->svm; // 支持向量机
            int pred = svm->eval(x); // pred = weights.transpose * x + bias
            if (pred > 0) ++count[pos_c]; // 预测值为正，则正类的票数加1
            else if (pred < 0) ++count[neg_c]; // 预测值为负，则负类的票数加1
        }

        // 选取票数最高的类别作为预测类别
        int max_value = -1; int max_pos = -1;
        for (int i = 0; i < class_num; ++i) {
            if (count[i] > max_value) {
                max_value = count[i];
                max_pos = i;
            }
        }

        ++confusion_matrix[y][max_pos];

        // 如果预测类别与标签值相同，则预测成功，否则预测失败
        if (max_pos == y) return 1;
        else return 0;
    }
};

class SupportVectorMachine_RBFKernal {
private:
    double bias; // 偏置
    int feature_dim_lower; // 映射前的低维特征向量的维度
    std::vector<double> alpha;

    double rbf_kernal(const std::vector<double>& x1,
        const std::vector<double>& x2, double sigma) {
        int dim = x1.size();
        double res = 0.0;
        for (int i = 0; i < dim; ++i)
            res += (x1[i] - x2[i]) * (x1[i] - x2[i]);
        return std::exp(-res / (sigma * sigma + sigma * sigma));
    }

    double compute_loss(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train, double sigma, const std::vector<double>& alpha0) {
        int n_train = x_train.size();
        double res = 0.0;
        for (int i = 0; i < n_train; ++i)
            for (int j = 0; j < n_train; ++j)
                res += alpha0[i] * alpha0[j] * y_train[i] * y_train[j]
                * rbf_kernal(x_train[i], x_train[j], sigma);
        res *= -0.5;
        double sum_alpha = 0.0;
        for (int i = 0; i < n_train; ++i)
            sum_alpha += alpha0[i];
        res += sum_alpha;
        return res;
    }

    std::vector<double> compute_grad(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train, double sigma, double C) {
        int n_train = x_train.size();
        std::vector<double> grad(n_train);

        for (int t = 0; t < n_train; ++t) {
            grad[t] = 1.0;
            for (int i = 0; i < n_train; ++i)
                grad[t] -= alpha[i] * rbf_kernal(x_train[i], x_train[t], sigma) * y_train[i] * y_train[t];

            // 控制alpha[t]在[0, C]上
            if (alpha[t] <= 0.0) grad[t] = std::max(0.0, grad[t]);
            else if (alpha[t] >= C) grad[t] = std::min(0.0, grad[t]);
        }

        return grad;
    }

public:
    SupportVectorMachine_RBFKernal(int feature_dim_lower) {
        this->feature_dim_lower = feature_dim_lower;
        bias = 0;
    }

    void train(const std::vector<std::vector<double>>& x_train,
        std::vector<int> y_train,
        const std::vector<std::vector<double>>& x_test,
        std::vector<int> y_test,
        int num_epochs, double learning_rate,
        double sigma, double init_alpha, double C) {

        int n_train = x_train.size(), n_test = x_test.size();

        for (int i = 0; i < n_train; ++i)
            if (y_train[i] == 0)
                y_train[i] = -1;
        for (int i = 0; i < n_test; ++i)
            if (y_test[i] == 0)
                y_test[i] = -1;

        alpha = std::vector<double>(n_train, init_alpha);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::vector<double> grad_alpha = compute_grad(x_train, y_train, sigma, C);

            for (int i = 0; i < n_train; ++i)
                alpha[i] += learning_rate * grad_alpha[i]; // 最大化，应当加学习率与梯度的乘积

            // 控制sum(y_train[i] * alpha[i]) = 0
            double sum = 0.0;
            for (int i = 0; i < n_train; ++i)
                sum += alpha[i] * y_train[i];
            if (sum != 0) {
                double avg_sum = sum / n_train;
                for (int i = 0; i < n_train; ++i)
                    alpha[i] -= avg_sum * y_train[i];
            }

            // 分别在训练集和测试集上进行预测
            // 求偏置
            struct AlphaWithNum {
                double alpha;
                int num;
            };
            std::vector<AlphaWithNum> alpha_with_num;
            for (int i = 0; i < n_train; ++i)
                alpha_with_num.push_back({ alpha[i], i });

            auto cmp = [&](AlphaWithNum alpha_with_num_1, AlphaWithNum alpha_with_num_2) -> bool {
                return alpha_with_num_1.alpha > alpha_with_num_2.alpha;
                };

            std::sort(alpha_with_num.begin(), alpha_with_num.end(), cmp); // 按照alpha_i的值降序排序

            double bias = 0.0;
            for (int i = 0; i < 10; ++i) { // 选择alpha_i的值最大的10个样本作为候选支持向量，求解bias
                int num = alpha_with_num[i].num;
                double bias_tmp = 0.0;
                for (int j = 0; j < n_train; ++j)
                    bias_tmp += alpha[j] * y_train[j] * rbf_kernal(x_train[j], x_train[num], sigma);
                bias_tmp = y_train[num] - bias_tmp;
                bias += bias_tmp;
            }
            bias /= 10.0; // 取平均

            // 在训练集上预测
            int cnt_correct_train = 0;
            for (int i = 0; i < n_train; ++i) {
                double y_pred = bias;
                for (int j = 0; j < n_train; ++j)
                    y_pred += alpha[j] * y_train[j] * rbf_kernal(x_train[j], x_train[i], sigma);
                if (y_pred > 0 && y_train[i] == 1 || y_pred < 0 && y_train[i] == -1)
                    ++cnt_correct_train;
            }
            double acc_train = 1.0 * cnt_correct_train / n_train;

            // 在测试集上预测
            int cnt_correct_test = 0;
            for (int i = 0; i < n_test; ++i) {
                double y_pred = bias;
                for (int j = 0; j < n_train; ++j)
                    y_pred += alpha[j] * y_train[j] * rbf_kernal(x_train[j], x_test[i], sigma);
                if (y_pred > 0 && y_test[i] == 1 || y_pred < 0 && y_test[i] == -1)
                    ++cnt_correct_test;
            }
            double acc_test = 1.0 * cnt_correct_test / n_test;

            std::cout << "[epoch " << epoch + 1 << "] accuracy_train_set = " << acc_train
                << ", accuracy_test_set = " << acc_test << "\n";
        }

        std::vector<double> y_pred_test;
        int cnt_correct_test = 0;
        for (int i = 0; i < n_test; ++i) {
            double y_pred = bias;
            for (int j = 0; j < n_train; ++j)
                y_pred += alpha[j] * y_train[j] * rbf_kernal(x_train[j], x_test[i], sigma);
            y_pred = 1.0 / (1.0 + std::exp(-y_pred));
            y_pred_test.push_back(y_pred);
        }

        // 绘制ROC曲线
        int pos_class_num = 0;
        int neg_class_num = 0;
        for (int i = 0; i < n_test; ++i) {
            if (y_test[i] == 1) ++pos_class_num;
            else ++neg_class_num;
        }

        struct RocDot {
            double false_positive_rate;
            double true_positive_rate;
        };

        std::vector<RocDot> roc;

        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.1) {
            int pos_pos_num = 0; // 正类样本被预测为正类的个数
            int neg_pos_num = 0; // 负类样本被预测为正类的个数
            for (int i = 0; i < n_test; ++i) {
                if (y_pred_test[i] >= threshold && y_test[i] == 1)
                    ++pos_pos_num;
                else if (y_pred_test[i] >= threshold && y_test[i] == 0)
                    ++neg_pos_num;
            }
            double fpr = 1.0 * neg_pos_num / neg_class_num;
            double tpr = 1.0 * pos_pos_num / pos_class_num;
            roc.push_back({ fpr, tpr });
        }

        std::cout << "\nROC:\n";
        for (auto roc_dot : roc)
            std::cout << roc_dot.false_positive_rate << " " << roc_dot.true_positive_rate << "\n";
    }

    void train_without_eval(const std::vector<std::vector<double>>& x_train,
        std::vector<int> y_train,
        int num_epochs, double learning_rate,
        double sigma, double init_alpha, double C) {

        int n_train = x_train.size();

        for (int i = 0; i < n_train; ++i)
            if (y_train[i] == 0)
                y_train[i] = -1;

        alpha = std::vector<double>(n_train, init_alpha);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::vector<double> grad_alpha = compute_grad(x_train, y_train, sigma, C);

            for (int i = 0; i < n_train; ++i)
                alpha[i] += learning_rate * grad_alpha[i]; // 最大化，应当加学习率与梯度的乘积

            // 控制sum(y_train[i] * alpha[i]) = 0
            double sum = 0.0;
            for (int i = 0; i < n_train; ++i)
                sum += alpha[i] * y_train[i];
            if (sum != 0) {
                double avg_sum = sum / n_train;
                for (int i = 0; i < n_train; ++i)
                    alpha[i] -= avg_sum * y_train[i];
            }

            // 分别在训练集和测试集上进行预测
            // 求偏置
            struct AlphaWithNum {
                double alpha;
                int num;
            };
            std::vector<AlphaWithNum> alpha_with_num;
            for (int i = 0; i < n_train; ++i)
                alpha_with_num.push_back({ alpha[i], i });

            auto cmp = [&](AlphaWithNum alpha_with_num_1, AlphaWithNum alpha_with_num_2) -> bool {
                return alpha_with_num_1.alpha > alpha_with_num_2.alpha;
                };

            std::sort(alpha_with_num.begin(), alpha_with_num.end(), cmp); // 按照alpha_i的值降序排序

            double bias = 0.0;
            for (int i = 0; i < 10; ++i) { // 选择alpha_i的值最大的10个样本作为候选支持向量，求解bias
                int num = alpha_with_num[i].num;
                double bias_tmp = 0.0;
                for (int j = 0; j < n_train; ++j)
                    bias_tmp += alpha[j] * y_train[j] * rbf_kernal(x_train[j], x_train[num], sigma);
                bias_tmp = y_train[num] - bias_tmp;
                bias += bias_tmp;
            }
            bias /= 10.0; // 取平均
        }

        // 在训练集上预测
        int cnt_correct_train = 0;
        for (int i = 0; i < n_train; ++i) {
            double y_pred = bias;
            for (int j = 0; j < n_train; ++j)
                y_pred += alpha[j] * y_train[j] * rbf_kernal(x_train[j], x_train[i], sigma);
            if (y_pred > 0 && y_train[i] == 1 || y_pred < 0 && y_train[i] == -1)
                ++cnt_correct_train;
        }
        double acc_train = 1.0 * cnt_correct_train / n_train;
    }

    double eval(const std::vector<double>& x, const std::vector<std::vector<double>>& x_train,
        std::vector<int> y_train, double sigma) {
        
        int n_train = x_train.size();
        
        for (int i = 0; i < n_train; ++i)
            if (y_train[i] == 0)
                y_train[i] = -1;
        
        double y_pred = bias;
        for (int j = 0; j < n_train; ++j)
            y_pred += alpha[j] * y_train[j] * rbf_kernal(x_train[j], x, sigma);

        return y_pred;
    }
};

class SupportVectorMachine_RBFKernal_multi_classes { // One-vs-One 方法实现多类分类
private:
    class Classifier {
    public:
        int positive_class; // 正类的编号
        int negative_class; // 负类的编号
        std::vector<std::vector<double>> x_train_2cls;
        std::vector<int> y_train_2cls;
        SupportVectorMachine_RBFKernal* svm; // 用于二分类的RBF核SVM

        Classifier(int positive_class, int negative_class, int feature_dim) {
            this->positive_class = positive_class;
            this->negative_class = negative_class;
            svm = new SupportVectorMachine_RBFKernal(feature_dim);
        }

        void set_x_train_2cls(std::vector<std::vector<double>> x_train_2cls) {
            this->x_train_2cls = x_train_2cls;
        }

        void set_y_train_2cls(std::vector<int> y_train_2cls) {
            this->y_train_2cls = y_train_2cls;
        }

        ~Classifier() { delete svm; }
    };

    std::vector<Classifier*> classifiers; // 存储所有分类器的数组
    int feature_dim; // 特征维度
    int class_num; // 类别个数
    int classifier_num; // 分类器的个数

public:
    SupportVectorMachine_RBFKernal_multi_classes(int feature_dim, int class_num) {
        this->feature_dim = feature_dim;
        this->class_num = class_num;
        classifier_num = (class_num & 1) ? ((class_num - 1) / 2 * class_num) : (class_num / 2 * (class_num - 1));
        for (int i = 0; i <= class_num - 2; ++i)
            for (int j = i + 1; j <= class_num - 1; ++j) {
                Classifier* c = new Classifier(i, j, feature_dim);
                classifiers.push_back(c);
            }
    }

    ~SupportVectorMachine_RBFKernal_multi_classes() {
        for (Classifier* classifier : classifiers)
            delete classifier;
    }

    void train(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train,
        const std::vector<std::vector<double>>& x_test,
        const std::vector<int>& y_test,
        int num_epochs, double learning_rate,
        double sigma, double init_alpha, double C) {

        for (auto classifier : classifiers) {
            int pos_c = classifier->positive_class;
            int neg_c = classifier->negative_class;
            SupportVectorMachine_RBFKernal* svm = classifier->svm;

            std::vector<std::vector<double>> x_train_2cls;
            std::vector<int> y_train_2cls;

            for (int i = 0; i < x_train.size(); ++i) {
                if (y_train[i] == pos_c) {
                    x_train_2cls.push_back(x_train[i]);
                    y_train_2cls.push_back(1);
                }
                else if (y_train[i] == neg_c) {
                    x_train_2cls.push_back(x_train[i]);
                    y_train_2cls.push_back(0);
                }
            }

            classifier->set_x_train_2cls(x_train_2cls);
            classifier->set_y_train_2cls(y_train_2cls);

            svm->train_without_eval(x_train_2cls, y_train_2cls, num_epochs, learning_rate, sigma, init_alpha, C);
        }

        int n_train = x_train.size();
        int cnt_correct_train = 0;
        for (int i = 0; i < n_train; ++i)
            cnt_correct_train += eval(x_train[i], y_train[i], sigma);
        double acc_train = 1.0 * cnt_correct_train / n_train;

        std::vector<std::vector<int>> confusion_matrix(class_num, std::vector<int>(class_num));

        int n_test = x_test.size();
        int cnt_correct_test = 0;
        for (int i = 0; i < n_test; ++i)
            cnt_correct_test += eval_test(x_test[i], y_test[i], sigma, confusion_matrix);
        double acc_test = 1.0 * cnt_correct_test / n_test;

        std::cout << "[epoch " << num_epochs << "] accuracy_train_set = "
            << acc_train << ", accuracy_test_set = " << acc_test << "\n";

        for (int i = 0; i < class_num; ++i) {
            for (int j = 0; j < class_num; ++j)
                std::cout << confusion_matrix[i][j] << " ";
            std::cout << "\n";
        }
    }

    int eval(const std::vector<double>& x, int y, double sigma) {
        std::vector<int> count(class_num);

        for (auto classifier : classifiers) {
            int pos_c = classifier->positive_class; // 正类编号
            int neg_c = classifier->negative_class; // 负类编号
            SupportVectorMachine_RBFKernal* svm = classifier->svm; // 支持向量机
            double pred = svm->eval(x, classifier->x_train_2cls, classifier->y_train_2cls, sigma); // pred = weights.transpose * x + bias
            if (pred > 0) ++count[pos_c]; // 预测值为正，则正类的票数加1
            else if (pred < 0) ++count[neg_c]; // 预测值为负，则负类的票数加1
        }

        // 选取票数最高的类别作为预测类别
        int max_value = -1; int max_pos = -1;
        for (int i = 0; i < class_num; ++i) {
            if (count[i] > max_value) {
                max_value = count[i];
                max_pos = i;
            }
        }

        // 如果预测类别与标签值相同，则预测成功，否则预测失败
        if (max_pos == y) return 1;
        else return 0;
    }

    int eval_test(const std::vector<double>& x, int y, double sigma,
        std::vector<std::vector<int>>& confusion_matrix) {

        std::vector<int> count(class_num);

        for (auto classifier : classifiers) {
            int pos_c = classifier->positive_class; // 正类编号
            int neg_c = classifier->negative_class; // 负类编号
            SupportVectorMachine_RBFKernal* svm = classifier->svm; // 支持向量机
            double pred = svm->eval(x, classifier->x_train_2cls, classifier->y_train_2cls, sigma); // pred = weights.transpose * x + bias
            if (pred > 0) ++count[pos_c]; // 预测值为正，则正类的票数加1
            else if (pred < 0) ++count[neg_c]; // 预测值为负，则负类的票数加1
        }

        // 选取票数最高的类别作为预测类别
        int max_value = -1; int max_pos = -1;
        for (int i = 0; i < class_num; ++i) {
            if (count[i] > max_value) {
                max_value = count[i];
                max_pos = i;
            }
        }

        ++confusion_matrix[y][max_pos];

        // 如果预测类别与标签值相同，则预测成功，否则预测失败
        if (max_pos == y) return 1;
        else return 0;
    }
};

class LogisticRegression_two_classes {
private:
    std::vector<double> param; // 模型参数（维度为feature_dim + 1）
    int feature_dim; // 原始特征维度

    std::vector<double> compute_grad(
        const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train
    ) {
        std::vector<double> res(feature_dim + 1);

        int n = x_train.size();
        for (int i = 0; i < n; ++i) {
            double tmp = 0.0;
            for (int j = 0; j < feature_dim + 1; ++j)
                tmp += x_train[i][j] * param[j];
            double factor = (1.0 / (1.0 + std::exp(-tmp)) - y_train[i]) / n;
            for (int j = 0; j < feature_dim + 1; ++j)
                res[j] += factor * x_train[i][j];
        }

        return res;
    }

public:
    LogisticRegression_two_classes(int feature_dim) {
        this->feature_dim = feature_dim;
        param = std::vector<double>(feature_dim + 1);
    }

    void train(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train,
        const std::vector<std::vector<double>>& x_test,
        const std::vector<int>& y_test,
        int num_epochs, double learning_rate) {

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::vector<double> grad = compute_grad(x_train, y_train);
            for (int i = 0; i < feature_dim + 1; ++i)
                param[i] -= learning_rate * grad[i];

            // 计算训练集和测试集的预测准确率
            int n_train = x_train.size();
            int cnt_correct_train = 0;
            for (int i = 0; i < n_train; ++i) {
                double y_pred = 0.0;
                for (int j = 0; j < feature_dim + 1; ++j)
                    y_pred += x_train[i][j] * param[j];
                if (y_pred > 0 && y_train[i] == 1 || y_pred < 0 && y_train[i] == 0)
                    ++cnt_correct_train;
            }
            double acc_train = 1.0 * cnt_correct_train / n_train;

            int n_test = x_test.size();
            int cnt_correct_test = 0;
            for (int i = 0; i < n_test; ++i) {
                double y_pred = 0.0;
                for (int j = 0; j < feature_dim + 1; ++j)
                    y_pred += x_test[i][j] * param[j];
                if (y_pred > 0 && y_test[i] == 1 || y_pred < 0 && y_test[i] == 0)
                    ++cnt_correct_test;
            }
            double acc_test = 1.0 * cnt_correct_test / n_test;

            std::cout << "[epoch " << epoch + 1 << "] accuracy_train_set = "
                << acc_train << ", accuracy_test_set = " << acc_test << "\n";
        }

        std::vector<double> y_pred_test;
        int n_test = x_test.size();
        int cnt_correct_test = 0;
        for (int i = 0; i < n_test; ++i) {
            double y_pred = 0.0;
            for (int j = 0; j < feature_dim + 1; ++j)
                y_pred += x_test[i][j] * param[j];
            y_pred = 1.0 / (1.0 + std::exp(-y_pred));
            y_pred_test.push_back(y_pred);
        }

        // 绘制ROC曲线
        int pos_class_num = 0;
        int neg_class_num = 0;
        for (int i = 0; i < n_test; ++i) {
            if (y_test[i] == 1) ++pos_class_num;
            else ++neg_class_num;
        }

        struct RocDot {
            double false_positive_rate;
            double true_positive_rate;
        };

        std::vector<RocDot> roc;

        for (double threshold = 0.1; threshold <= 0.9; threshold += 0.1) {
            int pos_pos_num = 0; // 正类样本被预测为正类的个数
            int neg_pos_num = 0; // 负类样本被预测为正类的个数
            for (int i = 0; i < n_test; ++i) {
                if (y_pred_test[i] >= threshold && y_test[i] == 1)
                    ++pos_pos_num;
                else if (y_pred_test[i] >= threshold && y_test[i] == 0)
                    ++neg_pos_num;
            }
            double fpr = 1.0 * neg_pos_num / neg_class_num;
            double tpr = 1.0 * pos_pos_num / pos_class_num;
            roc.push_back({ fpr, tpr });
        }

        std::cout << "\nROC:\n";
        for (auto roc_dot : roc)
            std::cout << roc_dot.false_positive_rate << " " << roc_dot.true_positive_rate << "\n";
    }
};

class LogisticRegression_multi_classes {
private:
    std::vector<std::vector<double>> param; // 模型参数：(feature_dim + 1) * class_num
    int feature_dim; // 原始特征维度
    int class_num; // 类别个数

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

    double compute_loss(
        const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train,
        const std::vector<std::vector<double>>& param0
    ) {
        std::vector<std::vector<double>> logits
            = matrix_multiply(x_train, param0);
        int n = x_train.size();
        double res = 0.0;
        for (int i = 0; i < n; ++i) {
            double all = 0.0;
            for (int j = 0; j < class_num; ++j)
                all += std::exp(logits[i][j]);
            res += logits[i][y_train[i]] - std::log(all);
        }
        res *= -1.0 / n;
        return res;
    }

    // 数值求导法计算梯度
    std::vector<std::vector<double>> compute_grad(
        const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train
    ) {
        std::vector<std::vector<double>> grad(feature_dim + 1, std::vector<double>(class_num));

        for (int i = 0; i < feature_dim + 1; ++i) {
            for (int j = 0; j < class_num; ++j) {
                std::vector<std::vector<double>> param1 = param;
                std::vector<std::vector<double>> param2 = param;
                param1[i][j] -= 1e-6;
                param2[i][j] += 1e-6;
                grad[i][j] = (compute_loss(x_train, y_train, param2)
                    - compute_loss(x_train, y_train, param1)) / 2e-6;
            }
        }

        return grad;
    }

    std::vector<int> argmax(const std::vector<std::vector<double>>& mat) {
        int m = mat.size();
        int n = mat[0].size();

        std::vector<int> res(m);

        for (int i = 0; i < m; ++i) {
            double max_value = -1e9;
            int pos = 0;
            for (int j = 0; j < n; ++j) {
                if (mat[i][j] > max_value) {
                    max_value = mat[i][j];
                    pos = j;
                }
            }
            res[i] = pos;
        }

        return res;
    }

public:
    LogisticRegression_multi_classes(int feature_dim, int class_num) {
        this->feature_dim = feature_dim;
        this->class_num = class_num;
        param = std::vector<std::vector<double>>(feature_dim + 1, std::vector<double>(class_num));
    }

    void train(const std::vector<std::vector<double>>& x_train,
        const std::vector<int>& y_train,
        const std::vector<std::vector<double>>& x_test,
        const std::vector<int>& y_test,
        int num_epochs, double learning_rate) {

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::vector<std::vector<double>> grad = compute_grad(x_train, y_train);
            for (int i = 0; i < feature_dim + 1; ++i)
                for (int j = 0; j < class_num; ++j)
                    param[i][j] -= learning_rate * grad[i][j];

            // 分别在训练集和测试集上评估
            std::vector<std::vector<int>> confusion_matrix(class_num, std::vector<int>(class_num));

            std::vector<int> y_pred_train = argmax(matrix_multiply(x_train, param));
            int n_train = y_pred_train.size();
            int cnt_correct_train = 0;
            for (int i = 0; i < n_train; ++i)
                if (y_pred_train[i] == y_train[i])
                    ++cnt_correct_train;
            double acc_train = 1.0 * cnt_correct_train / n_train;

            std::vector<int> y_pred_test = argmax(matrix_multiply(x_test, param));
            int n_test = y_pred_test.size();
            int cnt_correct_test = 0;
            for (int i = 0; i < n_test; ++i) {
                ++confusion_matrix[y_test[i]][y_pred_test[i]];
                if (y_pred_test[i] == y_test[i])
                    ++cnt_correct_test;
            }
            double acc_test = 1.0 * cnt_correct_test / n_test;

            double loss = compute_loss(x_train, y_train, param);

            std::cout << "[epoch " << epoch + 1 << "] " << "loss_train = "
                << loss << ", accuracy_train_set = "
                << acc_train << ", accuracy_test_set = " << acc_test << "\n";

            if (epoch + 1 == num_epochs) {
                for (int i = 0; i < class_num; ++i) {
                    for (int j = 0; j < class_num; ++j)
                        std::cout << confusion_matrix[i][j] << " ";
                    std::cout << "\n";
                }
            }

            if (acc_test > 0.95) learning_rate = 0.1;
        }
    }
};

int main() {
    // 二分类数据集
    std::cout << "二分类：\n\n";
    Dataset dataset_two_classes("dataset1.txt", 0.7, 683, 10, true);

    // 支持向量机
    std::cout << "支持向量机：\n\n";
    SupportVectorMachine svm_two_classes(10);
    svm_two_classes.train(dataset_two_classes.get_x_train(), dataset_two_classes.get_y_train(),
        dataset_two_classes.get_x_test(), dataset_two_classes.get_y_test(), 100, 1e-3, 20.0);

    // RBF核SVM
    std::cout << "\n\nRBF Kernal SVM:\n\n";
    SupportVectorMachine_RBFKernal svm_two_classes_rbf(10);
    svm_two_classes_rbf.train(dataset_two_classes.get_x_train(), dataset_two_classes.get_y_train(),
        dataset_two_classes.get_x_test(), dataset_two_classes.get_y_test(), 300, 1e-2, 1.0, 0.0, 5.0);

    // 逻辑回归
    std::cout << "\n\n逻辑回归：\n\n";
    LogisticRegression_two_classes lr_two_classes(10);
    lr_two_classes.train(dataset_two_classes.get_x_train(), dataset_two_classes.get_y_train(),
        dataset_two_classes.get_x_test(), dataset_two_classes.get_y_test(), 1000, 0.1);

    std::cout << "\n\n\n\n\n";
    
    // 多分类数据集
    std::cout << "多分类：\n\n";
    Dataset dataset_multi_classes("dataset2.txt", 0.5, 150, 4, false);

    // 支持向量机
    std::cout << "支持向量机：\n\n";
    SupportVectorMachine_multi_classes svm_multi_classes(4, 3);
    svm_multi_classes.train(dataset_multi_classes.get_x_train(), dataset_multi_classes.get_y_train(),
        dataset_multi_classes.get_x_test(), dataset_multi_classes.get_y_test(), 1000, 1e-2, 150.0);

    // RBF核SVM
    std::cout << "\n\nRBF Kernal SVM:\n\n";
    SupportVectorMachine_RBFKernal_multi_classes svm_multi_classes_rbf(4, 3);
    svm_multi_classes_rbf.train(dataset_multi_classes.get_x_train(), dataset_multi_classes.get_y_train(),
        dataset_multi_classes.get_x_test(), dataset_multi_classes.get_y_test(), 1000, 1e-2, 1.0, 0.0, 50.0);
        
    // 逻辑回归
    std::cout << "\n\n逻辑回归：\n\n";
    LogisticRegression_multi_classes lr_multi_classes(4, 3);
    lr_multi_classes.train(dataset_multi_classes.get_x_train(), dataset_multi_classes.get_y_train(),
        dataset_multi_classes.get_x_test(), dataset_multi_classes.get_y_test(), 1000, 1.0);
        
    return 0;
}