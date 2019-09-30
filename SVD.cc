/*
  Author     : Maksim Y. Melnichenko
  Course     : Numerical Analysis
  Assignment : 09-Linear System

/************************************************************/
// System includes

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <stdio.h>
#include <utility>
#include <vector>

/************************************************************/
// Using declarations

using std::cin;
using std::cout;
using std::endl;
using std::pow;
using std::vector;
using namespace std;

template <bool Transpose = false, bool Owning = true> struct MatrixT {
private:
  int rows_;
  int colms_;
  double *data_;

public:
  double &operator()(int r, int c) {
    if constexpr (Transpose) {
      return data_[c * colms_ + r];
    } else {
      return data_[r * colms_ + c];
    }
  }

  double operator()(int r, int c) const {
    if constexpr (Transpose) {
      return data_[c * colms_ + r];
    } else {
      return data_[r * colms_ + c];
    }
  }

  MatrixT<!Transpose, false> &transpose() const {
    return *reinterpret_cast<MatrixT<!Transpose, false> *>(
        const_cast<MatrixT<Transpose, Owning> *>(this));
  }

  int rows() const {
    if constexpr (Transpose) {
      return colms_;
    } else {
      return rows_;
    }
  }

  int colms() const {
    if constexpr (Transpose) {
      return rows_;
    } else {
      return colms_;
    }
  }

  MatrixT(int r, int c) : rows_(r), colms_(c), data_(nullptr) {
    data_ = new double[c * r];
  }

  ~MatrixT() {
    if constexpr (Owning) {
      delete[] data_;
    }
  }

  template <bool Trans, bool Own,
            typename = std::enable_if_t<(Trans ^ Transpose) | (Own ^ Owning)>>
  MatrixT(MatrixT<Trans, Own> const &other)
      : rows_(other.rows_), colms_(other.colms_), data_(other.data_) {
    if constexpr (Owning) {
      data_ = new double[rows_ * colms_];
      std::copy_n(other.data_, rows_ * colms_, data_);
    }
  }

  MatrixT(const MatrixT &other)
      : rows_(other.rows_), colms_(other.colms_), data_(other.data_) {
    if constexpr (Owning) {
      data_ = new double[rows_ * colms_];
      std::copy_n(other.data_, rows_ * colms_, data_);
    }
  }

  MatrixT(MatrixT &&other)
      : rows_(std::exchange(other.rows_, 0)),
        colms_(std::exchange(other.colms_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}

  template <bool Own> MatrixT &operator=(const MatrixT<Transpose, Own> &other) {
    if (&other != this) {
      MatrixT<Transpose, true> copy(other);
      std::swap(rows_, copy.rows_);
      std::swap(colms_, copy.cols_);
      if constexpr (Owning) {
        delete[] std::exchange(data_, nullptr);
      }
      std::swap(data_, copy.data_);
    }
    return *this;
  }

  MatrixT &operator=(MatrixT &&other) {
    std::swap(rows_, other.rows_);
    std::swap(colms_, other.colms_);
    if constexpr (Owning)
      delete[] std::exchange(data_, nullptr);
    std::swap(data_, other.data_);
    return *this;
  }

  template <bool T1, bool O1>
  static MatrixT cut(const MatrixT<T1, O1> &origin, int stitch, bool up) {
    // dimension of symmetric & region
    int m, s, b;

    if (up) {
      m = b = stitch;
      s = 0;
    } else {
      m = origin.rows() - stitch;
      s = stitch;
      b = origin.rows();
    }

    MatrixT piece(m, m);

    for (s; s < b; ++s) {
      for (int cs = s; cs < b; ++cs) {
        piece(s, cs) = origin(s, cs);
      }
    }

    return piece;
  }
};

using Matrix = MatrixT<>;

template <bool T1, bool T2, bool O1, bool O2>
Matrix operator*(MatrixT<T1, O1> const &a, MatrixT<T2, O2> const &b) {
  if (a.colms() != b.rows()) {
    throw std::runtime_error("This does not work.");
  }
  Matrix res(a.rows(), b.colms());

  for (int i = 0; i < a.rows(); ++i) {
    for (int j = 0; j < b.colms(); ++j) {
      for (int k = 0; k < a.colms(); ++k) {
        res(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return res;
}

/************************************************************/
// Function prototypes/global vars/typedefs

void build(Matrix &A);

void tridiagonalizer(Matrix &B);

void similarityProducer(Matrix &WTW, double RSQ);

template <bool T1, bool O1, bool T2, bool O2>
void block_diagonal(MatrixT<T1, O1> &B, MatrixT<T2, O2> &Beta);

void divide(Matrix &B);

void conquer();

void printMatrix(Matrix A);

/************************************************************/
// Main function

int main(int argc, char *argv[]) {

  // int r = 2;
  // int c = 4;

  // if ( r < c)
  //{
  //    int i = r;
  //    r = c;
  //    c = i;
  //}
  // Since (A^T) * A == A * (A^T)

  // Matrix A (r, c);

  // build(A);
  // Initial Matrix
  // printMatrix(A);

  // auto B = A * A.transpose();

  // Building a numerical example

  Matrix TEST(4, 4);

  TEST(0, 0) = 4;
  TEST(0, 1) = 1;
  TEST(0, 2) = -2;
  TEST(0, 3) = 2;

  TEST(1, 0) = 1;
  TEST(1, 1) = 2;
  TEST(1, 2) = 0;
  TEST(1, 3) = 1;

  TEST(2, 0) = -2;
  TEST(2, 1) = 0;
  TEST(2, 2) = 3;
  TEST(2, 3) = -2;

  TEST(3, 0) = 2;
  TEST(3, 1) = 1;
  TEST(3, 2) = -2;
  TEST(3, 3) = -1;

  // A * (A ^ T)
  // printMatrix(B);

  tridiagonalizer(TEST);
  // After Householder's Transformation
  printMatrix(TEST);
}

void build(Matrix &A) {
  int i, j;
  for (i = 0; i < A.rows(); ++i) {
    for (j = 0; j < A.colms(); ++j) {
      A(i, j) = rand() % 100;
    }
  }
}

void tridiagonalizer(Matrix &B) {
  // producing a column vector, replicating the last column of B

  for (int k = 0; k < B.rows() - 2; ++k) {
    double alpha = 0;
    double RSQ = 0;
    Matrix W(B.rows(), k + 1);
    // for k = 0 ...  < n-2

    for (int i = k + 1; i < B.rows(); ++i) {
      alpha += std::pow(B(i, k), 2);
      W(i, k) = (0.5) * B(i, k);
    }

    alpha = -(std::sqrt(alpha));
    RSQ = (std::pow(alpha, 2) - (alpha * B(k + 1, k)));

    W(k, k) = 0;
    W(k + 1, k) = (0.5) * (B(k + 1, k) - alpha);

    auto WTW = W * W.transpose();

    similarityProducer(WTW, RSQ);

    cout << k << "th Similarity Matrix:\n\n";
    printMatrix(WTW);

    B = WTW * B * WTW;
    printMatrix(B);
  }
}

void divide(Matrix &B) {
  Matrix Beta(B.rows(), B.colms());
  block_diagonal(B, Beta);

  if (B.rows() < 2) {
    // return
  } else {
    Matrix hi = Matrix::cut(B, B.rows() / 2, 1);
    Matrix lo = Matrix::cut(B, B.colms() - (B.colms() / 2), 0);
    // return Beta + [(divide (hi))    (divide (lo))]
  }
}

void similarityProducer(Matrix &WTW, double RSQ) {
  for (int i = 0; i < WTW.rows(); ++i) {
    for (int j = 0; j < WTW.colms(); ++j) {
      WTW(i, j) = -(4 / RSQ) * WTW(i, j);
    }
    WTW(i, i) += 1;
  }
}

template <bool T1, bool O1, bool T2, bool O2>
void block_diagonal(MatrixT<T1, O1> &B, MatrixT<T2, O2> &Beta) {

  double m = B.rows() / 2;
  double beta_value = B(m, m - 1);

  Beta(m, m) = Beta(m - 1, m - 1) = Beta(m, m - 1) = Beta(m - 1, m) =
      beta_value;

  B(m, m - 1) = B(m - 1, m) = 0;
  B(m, m) -= beta_value;
  B(m - 1, m - 1) -= beta_value;
}

void printMatrix(const Matrix A) {
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.colms(); ++j) {
      printf("%13.5f", A(i, j));
    }
    puts("");
  }
  puts("");
}
