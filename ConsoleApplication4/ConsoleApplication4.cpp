#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

// Structure to store test case
struct TestCase {
    MatrixXd A;
    VectorXd b;
    string description;
};

int main() {
    vector<TestCase> tests;

    // Example 1: Overdetermined inconsistent
    tests.push_back({
        (MatrixXd(3, 2) << 1, 2,
                           2, 4,
                           3, 5).finished(),
        (VectorXd(3) << 1, 2, 2).finished(),
        "Overdetermined inconsistent system"
        });

    // Example 2: Underdetermined consistent
    tests.push_back({
        (MatrixXd(2, 3) << 1, 0, 1,
                           0, 1, 1).finished(),
        (VectorXd(2) << 3, 3).finished(),
        "Underdetermined consistent system"
        });

    // Example 3: Overdetermined inconsistent
    tests.push_back({
        (MatrixXd(4, 2) << 1, 1,
                           1, 2,
                           1, 3,
                           1, 4).finished(),
        (VectorXd(4) << 6, 5, 7, 10).finished(),
        "Overdetermined inconsistent system #2"
        });

    // Example 4: Square and consistent
    tests.push_back({
        (MatrixXd(2, 2) << 2, 1,
                           1, 3).finished(),
        (VectorXd(2) << 4, 5).finished(),
        "Square consistent system"
        });

    // Example 5: Underdetermined consistent with free variables
    tests.push_back({
        (MatrixXd(2, 3) << 1, 2, 1,
                           0, 1, 1).finished(),
        (VectorXd(2) << 3, 2).finished(),
        "Underdetermined consistent system #2"
        });

    for (size_t i = 0; i < tests.size(); ++i) {
        cout << "=== Test Case #" << i + 1 << ": " << tests[i].description << " ===\n\n";

        MatrixXd A = tests[i].A;
        VectorXd b = tests[i].b;

        // Compute pseudo-inverse using SVD
        JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
        VectorXd singular_values = svd.singularValues();

        VectorXd singular_values_inv(singular_values.size());
        const double tolerance = 1e-6;
        for (int i = 0; i < singular_values.size(); ++i) {
            singular_values_inv(i) = (singular_values(i) > tolerance) ? 1.0 / singular_values(i) : 0.0;
        }

        MatrixXd S_inv = singular_values_inv.asDiagonal();
        MatrixXd A_pseudo = svd.matrixV() * S_inv * svd.matrixU().transpose();

        VectorXd x = A_pseudo * b;
        VectorXd r = A * x - b;
        double error = r.norm();

        // Output
        cout << "A:\n" << A << "\n\n";
        cout << "b:\n" << b.transpose() << "\n\n";
        cout << "x (solution using pseudo-inverse):\n" << x.transpose() << "\n\n";
        cout << "||r||2 (Euclidean error): " << error << "\n\n";
    }

    return 0;
}
