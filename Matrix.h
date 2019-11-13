#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
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

  using iterator = double*;
  using const_iterator = const double*;
  using iterator_category = std::random_access_iterator_tag;
  using value_type = double;
  using reference = value_type&;
  using pointer = value_type*;
  using const_reference = const value_type&;
  using const_pointer = const value_type*;

  using default_type = MatrixT<false, true>;

  iterator begin() {
    return data_;
  }

  const_iterator begin() const {
    return data_;
  }

  const_iterator cbegin() const {
    return data_;
  }

  size_t size() const {
    return (size_t)rows_ * colms_;
  }

  iterator end() {
    return begin() + size();
  }

  const_iterator end() const {
    return begin() + size();
  }

  const_iterator cend() const {
    return begin() + size();
  }

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
    std::fill_n(data_, c * r, 0.0);
  }

  ~MatrixT() {
    if constexpr (Owning) {
      delete[] data_;
    }
  }

//

  template <bool Trans, bool Own,
            typename = std::enable_if_t<(Trans ^ Transpose) | (Own ^ Owning)>>
  MatrixT(MatrixT<Trans, Own> const &other)
      : rows_(other.rows_), colms_(other.colms_), data_(other.data_) {
    if constexpr (Owning) {
      data_ = new double[rows_ * colms_];
    }
    if constexpr (Transpose ^ Trans) {
      for (int r = 0; r < other.rows(); ++r) {
        for (int c = 0; c < other.colms(); ++c) {
          (*this) (c, r) = other(r, c);
        }
      }
    } else {
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

//

  MatrixT &operator=(MatrixT &&other) {
    std::swap(rows_, other.rows_);
    std::swap(colms_, other.colms_);
    if constexpr (Owning)
      delete[] std::exchange(data_, nullptr);
    std::swap(data_, other.data_);
    return *this;
  }

  template <typename Scalar>
  std::enable_if_t<std::is_arithmetic_v<Scalar>, MatrixT&>
  operator*= (Scalar s) {
    for (auto & v : *this) {
      v *= s;
    }
    return *this;
  }

  default_type cut(int stitch, bool upper) const {
    int originRows = 0;
    int originCols = 0;
    int rows = stitch;
    int cols = stitch;

    if (!upper) {
      originRows = rows;
      originCols = cols;
      rows = this->rows() - rows;
      cols = this->colms() - cols;
    }

    default_type piece(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        piece(i, j) = (*this)(originRows + i, originCols + j);
      }
    }
    return piece;
  }

  template <bool T1, bool O1, bool T2, bool O2>
  static default_type combine(const MatrixT<T1, O1> &hi, const MatrixT<T2, O2> &lo) {

    int a = hi.rows();
    int b = hi.colms();
    int c = lo.rows();
    int d = lo.colms();

    default_type res(a + c, b + d);

    for(int i = 0; i < a; ++i)
    {
      for(int j = 0; j < b; ++j)
      {
        res(i, j) = hi(i, j);
      }
    }

    for(int i = 0; i < c; ++i)
    {
      for(int j = 0; j < d; ++j)
      {
        res(i + a, j + b) = lo(i, j);
      }
    }

    return res;
  }

  template <bool T1, bool O1>
  static default_type diagSort(const MatrixT<T1, O1> &diag)
  {
    int n = diag.rows();
    default_type D (n, n);
    std::vector<double> buf (n);
    
    for(int i = 0; i < n; ++ i)
    {
      buf[i] = diag(i, i);
    }

    std::sort(buf.begin(), buf.end());

    for(int i = 0; i < n; ++i)
    {
      D(i, i) = buf[i];
    }

    return D;
  }

  static default_type identity(const int size) {

    default_type result {size, size};

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

  int ar = a.rows();
  int ac = a.colms();
  int br = b.rows();
  int bc = b.colms();

  if (ac != br) {
    throw std::runtime_error("Illegal dimensions given. Please, adhere to the formal rules");
  }
  Matrix res(ar, bc);

  for (int i = 0; i < ar; ++i) {
    for (int k = 0; k < ac; ++k) {
      for (int j = 0; j < bc; ++j) {
        res(i, j) += a(i, k) * b(k, j);
      }
    }
  }
  return res;
}


template <bool T1, bool O1, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>> >
Matrix operator*(const MatrixT<T1, O1> &M, Scalar a) {
  Matrix A (M);
  for (auto & val : A) {
    val *= a;
  }
  return A;
}

template <bool T1, bool O1, typename Scalar, typename = std::enable_if_t<std::is_arithmetic_v<Scalar>>>
Matrix operator*(Scalar a, const MatrixT<T1, O1> &M) {
  Matrix A (M);
  for (auto & val : A) {
    val *= a;
  }
  return A;
}


template <bool T1, bool T2, bool O1, bool O2>
Matrix operator+(MatrixT<T1, O1> const &a, MatrixT<T2, O2> const &b) {

  int ar = a.rows();
  int ac = a.colms();
  int br = b.rows();
  int bc = b.colms();

  if ((ar != br) && (ac != bc)) {
    throw std::runtime_error("Illegal dimensions given. Please, adhere to the formal rules");
  }


Matrix res(ar, bc);

  for (int i = 0; i < ar; ++i) {
    for (int j = 0; j < ac; ++j) {
        res(i, j) = a(i, j) + b(i, j);
      }
    }

  return res;
}


#endif
