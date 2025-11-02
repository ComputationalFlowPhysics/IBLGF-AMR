#include <iostream>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>

// LAPACK prototype for dgees
extern "C" {
    void dgees_(
        char* jobvs, char* sort, int (*select)(double*, double*),
        int* n, double* a, int* lda, int* sdim,
        double* wr, double* wi, double* vs, int* ldvs,
        double* work, int* lwork, int* bwork, int* info
    );
}

// Example selection function: keep eigenvalues with Re(λ) < 0 in top-left block
extern "C" int select_eig(double* wr, double* wi) {
    return (*wr < 0.0);
}

int main() {
    using namespace xt;

    // Example matrix (column-major convention for LAPACK)
    xarray<double> A = {
        {1.0, 2.0, 3.0},
        {0.0, 4.0, 5.0},
        {0.0, 0.0, -2.0}
    };

    int n = static_cast<int>(A.shape()[0]);
    int lda = n;
    int sdim = 0, info = 0;
    char jobvs = 'V';  // Compute Schur vectors Q
    char sort = 'S';   // 'S' = sort eigenvalues, 'N' = no sorting

    std::vector<double> wr(n), wi(n);       // Eigenvalue real/imag parts
    std::vector<double> vs(n * n);          // Schur vectors (Q)
    std::vector<int> bwork(n);
    std::vector<double> work(1);
    int lwork = -1; // workspace query

    // Workspace query
    dgees_(&jobvs, &sort, select_eig, &n, A.data(), &lda, &sdim,
           wr.data(), wi.data(), vs.data(), &lda,
           work.data(), &lwork, bwork.data(), &info);

    lwork = static_cast<int>(work[0]);
    work.resize(lwork);

    // Actual computation
    dgees_(&jobvs, &sort, select_eig, &n, A.data(), &lda, &sdim,
           wr.data(), wi.data(), vs.data(), &lda,
           work.data(), &lwork, bwork.data(), &info);

    if (info != 0) {
        std::cerr << "dgees failed with info = " << info << std::endl;
        return 1;
    }

    // Output results
    std::cout << "Ordered Real Schur form T:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
        std::cout << "\n";
    }

    std::cout << "\nOrthogonal matrix Q:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            std::cout << vs[i + j * n] << " ";
        std::cout << "\n";
    }

    std::cout << "\nEigenvalues (Re, Im):\n";
    for (int i = 0; i < n; ++i)
        std::cout << "(" << wr[i] << ", " << wi[i] << ")\n";

    std::cout << "\nNumber of selected eigenvalues (Re < 0): " << sdim << "\n";
    return 0;
}
