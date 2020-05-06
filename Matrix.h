/*
Matrix.h - header file, containing the Matrix structure, used in the SVD.cc file.
*/


#ifndef MATRIX_H_
#define MATRIX_H_

#if !defined(SVD_MAX_OMP_THREADS)
#define SVD_MAX_OMP_THREADS 4
#endif

#if !defined(SVD_MAX_PAR_DEPTH)
#define SVD_MAX_PAR_DEPTH 2
#endif

/************************************************************/
// System includes
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <omp.h>


template <bool Transpose = false, bool Owning = true> struct MatrixT {

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
  using MatrixPair = std::pair<MatrixT, MatrixT>;

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

  template <bool Trans, bool Own>
           // typename = std::enable_if_t<(Trans ^ Transpose) | (Own ^ Owning)>>
  MatrixT(MatrixT<Trans, Own> const &other)
      : rows_(other.rows()), colms_(other.colms()), data_(const_cast<double*>(&*other.begin())) {
    if constexpr (Owning) {
      data_ = new double[size()];
    }
    if constexpr (Transpose ^ Trans) {
      for (int r = 0; r < other.rows(); ++r) {
        for (int c = 0; c < other.colms(); ++c) {
          (*this) (c, r) = other(r, c);
        }
      }
    } else {
      std::copy_n(other.begin(), size(), begin());
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

  template <bool Trans, bool Own>
           // typename = std::enable_if_t<(Trans ^ Transpose) | (Own ^ Owning)>>
  MatrixT(MatrixT<Trans, Own> &&other)
      : rows_(std::exchange(other.rows_, 0)),
        colms_(std::exchange(other.colms_, 0)),
        data_(std::exchange(other.data_, nullptr)) {
  }

   MatrixT &operator=(const MatrixT &other) {
    if (&other != this) {
      MatrixT copy(other);
      std::swap(rows_, copy.rows_);
      std::swap(colms_, copy.colms_);
      std::swap(data_, copy.data_);
    }
    return *this;
  }

  template <bool Trans, bool Own> MatrixT &operator=(const MatrixT<Trans, Own> &other) {
    MatrixT<Transpose, true> copy(other);
    std::swap(rows_, copy.rows_);
    std::swap(colms_, copy.colms_);
    if constexpr (Owning) {
      delete[] std::exchange(data_, nullptr);
    }
    std::swap(data_, copy.data_);
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

  /*
  column_extract - retrieving a column from a matrix at a specified index.
  */

  default_type column_extract(int pos) 
  {
    const int n = this -> rows();

    default_type res (n, 1);
    
    for(int i = 0; i < n; ++i)
    {
      res(i, 0) = (*this)(i, pos);
      //(*this)(i, pos) = 0;
    }

    return res;
  }

  /*
  column_immerse - placing the column at the given index.
  */

  //C for column, D for Destination
  //Exception would be nice
  
  static default_type column_immerse(const MatrixT &C, MatrixT &D,  int pos) 
  {
    const int n = D.rows();
    
    for(int i = 0; i < n; ++i)
    {
      D(i, pos) = C(i, 0);
    }

    return D;
  }

  /*
  column_extract - retrieving a column from a matrix at a specified index.
  */

  default_type row_extract( int pos) 
  {
    const int m = this->colms();

    default_type res (1, m);
    
    for(int i = 0; i < m; ++i)
    {
      res(0, i) = (*this)(pos, i);
      //(*this)(i, pos) = 0;
    }

    return res;
  }

  /*
  column_immerse - placing the column at the given index.
  */

  //C for column, D for Destination
  //Exception would be nice
  
  static default_type row_immerse(const MatrixT &R, MatrixT &D,  int pos) 
  {
    int m = D.colms();
    
    for(int i = 0; i < m; ++i)
    {
      D(pos, i) = R(0, i);
    }

    return D;
  }


  /*
  cut - intended to cut a block diagonal matrix into the upper and the lower matrix
  Is used in DnC algorithm.
  */

  default_type cut(int stitch, bool upper) const 
  {
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

  /*
  combine - combines two separate matrices, designated as "hi" and "lo" into one
  block diagonal matrix. Is used in DnC algorithm.
  */

  template <bool T1, bool O1, bool T2, bool O2>
  static default_type combine(const MatrixT<T1, O1> &hi, const MatrixT<T2, O2> &lo) 
  {

    const int a = hi.rows();
    const int b = hi.colms();
    const int c = lo.rows();
    const int d = lo.colms();

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

  /*
  magnitude - returns a magnitude of a specified vector;
  We are ONLY dealing with column vectors.
  CAN GET RID OF
  */

  
  static double magnitude(const MatrixT &V)
  {

    const int n = V.rows();
    double magnitude = 0;

    for (int i = 0; i < n; ++i)
    {
      double cell = V(i, 0);
      magnitude += cell * cell;
    }

    return std::sqrt(magnitude);
  }

  /*
  identity - returns an identity matrix of a given size.
  */
  static default_type identity(const int size) {

    default_type result {size, size};

    for( int i = 0; i < size; ++i)
    {
      result (i, i) = 1;
    }
    return result;
  }

};


/*
Below - Matrix multiplication and addition definitions.
*/
using Matrix = MatrixT<>;

template <bool T1, bool T2, bool O1, bool O2>
Matrix operator*(MatrixT<T1, O1> const &A, MatrixT<T2, O2> const &B) {

  int ar = A.rows();
  int ac = A.colms();
  int br = B.rows();
  int bc = B.colms();

  const int blockSize = 8;

  if (ac != br) {
    throw std::runtime_error("Illegal dimensions given. Please, adhere to the formal rules");
  }
  Matrix res(ar, bc);
  
  // TODO -- determine best loop nesting (parallel)
  /*for (int i = 0; i < ar; ++i) {
    for (int k = 0; k < ac; ++k) {
      for (int j = 0; j < bc; ++j) {
        res(i, j) += A(i, k) * B(k, j);
      }
    }
  }*/
  
  #pragma omp parallel for 
  for (int jj = 0; jj < bc; jj += blockSize) 
  {
      for (int kk = 0; kk < ac; kk += blockSize)
      {
          for (int i = 0; i < ar; ++i) 
          {
              #pragma omp parallel for 
              for (int j = jj; j < std::min(jj + blockSize , bc); ++j) 
              {
                  double sum = 0.0;
                  #pragma omp parallel for num_threads(4) reduction(+ : sum)
                  for (int k = kk; k < std::min(kk + blockSize, ac); ++k) 
                  {
                      sum += A(i, k) * B(k, j);
                  }
                  res(i, j) += sum;
              }
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

  const int ar = a.rows();
  const int ac = a.colms();
  const int br = b.rows();
  const int bc = b.colms();

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

#pragma omp declare reduction (mat_add : Matrix : omp_out = omp_out + omp_in) initializer (omp_priv = omp_orig)

template <bool T1, bool T2, bool O1, bool O2>
Matrix operator-(MatrixT<T1, O1> const &a, MatrixT<T2, O2> const &b) {

  const int ar = a.rows();
  const int ac = a.colms();
  const int br = b.rows();
  const int bc = b.colms();

  if ((ar != br) && (ac != bc)) {
    throw std::runtime_error("Illegal dimensions given. Please, adhere to the formal rules");
  }


Matrix res(ar, bc);

  for (int i = 0; i < ar; ++i) {
    for (int j = 0; j < ac; ++j) {
        res(i, j) = a(i, j) - b(i, j);
      }
    }

  return res;
}


#pragma omp declare reduction (mat_sub : MatrixT<true,true> : omp_out = omp_out + omp_in) initializer (omp_priv = omp_orig)
#pragma omp declare reduction (mat_sub : MatrixT<true,false> : omp_out = omp_out + omp_in) initializer (omp_priv = omp_orig)
#pragma omp declare reduction (mat_sub : MatrixT<false,false> : omp_out = omp_out + omp_in) initializer (omp_priv = omp_orig)
#pragma omp declare reduction (mat_sub : MatrixT<false,true> : omp_out = omp_out + omp_in) initializer (omp_priv = omp_orig)

#endif
