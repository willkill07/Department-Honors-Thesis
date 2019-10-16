#ifndef MATRIX_H_
#define MATRIX_H_


#include <algorithm>
#include <utility>
#include <type_traits>
#include <stdexcept>
/*
Header file, containing the matix struct.
*/

/************************************************************/
// System includes

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

  static MatrixT identity(const int size) {
    MatrixT result {size, size};
    for( int i = 0; i < size; ++i)
    {
      result (i, i) = 1;
    }
    return result;
  }
};

using Matrix = MatrixT<>;

template <bool T1, bool T2, bool O1, bool O2>
Matrix operator*(MatrixT<T1, O1> const &a, MatrixT<T2, O2> const &b) {
  if (a.colms() != b.rows()) {
    throw std::runtime_error("Illegal dimensions given. Please, adhere to the formal rules");
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

#endif