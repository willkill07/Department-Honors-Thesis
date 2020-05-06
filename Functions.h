#ifndef FUNCTIONS_H_  
#define FUNCTIONS_H_  

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <random>
#include <stdexcept>
#include <stdio.h>
#include <utility>
#include <vector>
#include <thread>
#include <future>

//custom matrix header
#include "Matrix.h"

using std::cin;
using std::cout;
using std::endl;
using std::pow;
using std::vector;
using std::sqrt;
using std::pow;

using Correction = std::pair<double, Matrix>;
using MatrixPair = std::pair<Matrix, Matrix>;
using MatrixTuple = std::tuple<Matrix, Matrix, Matrix>;

/************************************************************/
// Small mathematical functions

template <bool T1, bool O1, bool T2, bool O2>
double secular_function(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double rho, const double y)
{
  const int m = Diag.rows();
  double sum = 0;
  double z = 0;

  //#pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < m; ++j)
  {
    z = Z(j, 0);
    sum += (z * z) / (Diag(j, j) - y);
  }

  return rho + sum;
}

template <bool T1, bool O1, bool T2, bool O2>
double secular_function_prime(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double y)
{
  const int m = Diag.rows();
  double sum = 0;
  double z = 0;
  double denom; //denominator

  //#pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < m; ++j)
  {
    z = Z(j, 0);
    denom = Diag(j, j) - y;
    sum += (z * z) / (denom * denom);
  }

  return sum;
}

template <bool T1, bool O1, bool T2, bool O2>
double psi(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double y, const int k)
{
  double sum = 0;
  double z = 0;

  //#pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < k; ++j)
  {
    z = Z(j, 0);
    sum += (z * z) / ( Diag(j, j) - y);
  }

  return sum;
}

template <bool T1, bool O1, bool T2, bool O2>
double phi(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double y, const int k)
{
  const int m = Diag.rows();
  double sum = 0;
  double z = 0;

  //#pragma omp parallel for reduction(+ : sum)
  for (int j = k; j < m; ++j)
  {
    z = Z(j, 0);
    sum += (z * z) / (Diag(j, j) - y);
  }

  return sum;
}

template <bool T1, bool O1, bool T2, bool O2>
double psi_prime(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double y, const int k)
{
  double sum = 0;
  double z = 0;
  double denom = 0; //denominator

  //#pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < k; ++j)
  {
    z = Z(j, 0);
    denom = Diag(j, j) - y;
    sum += (z * z) / (denom * denom);
  }

  return sum;
}

template <bool T1, bool O1, bool T2, bool O2>
double phi_prime(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double y, const int k)
{
  const int m = Diag.rows();
  double sum = 0;
  double z = 0;
  double denom = 0; //denominator

  //#pragma omp parallel for reduction(+ : sum)
  for (int j = k; j < m; ++j)
  {
    z = Z(j, 0);
    denom = Diag(j, j) - y;
    sum += (z * z) / (denom * denom);
  }

  return sum;
}

template <bool T1, bool O1, bool T2, bool O2>
double g_function(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double rho, const double y, const int k)
{
  const int m = Diag.rows();
  double sum = 0;
  double z = 0;

  //#pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < m; ++j)
  {
    if (m != k && m != (k + 1))
    {
      z = Z(j, 0);
      sum += (z * z) / (Diag(j, j) - y);
    }
  }

  return rho + sum;
}

template <bool T1, bool O1, bool T2, bool O2>
double h_function(const MatrixT<T1, O1>& Diag, const MatrixT<T2, O2>& Z, const double y, const int k)
{
  double z1 = Z(k, 0);
  double z2 = Z(k + 1, 0);
  double sum1 = ((z1 * z1) / (Diag(k, k) - y));
  double sum2 = ((z2 * z2) / (Diag(k + 1, k + 1) - y));
  cout << "here " << Diag(k + 1, k + 1) << "\n";
  cout << "here " << Diag(k, k) << "\n";

  return sum1 + sum2;
}
/************************************************************/

/************************************************************/
// Helper function prototypes

template <bool O, bool T>
void print_matrix(const MatrixT<O,T> &A)
{
  for (int i = 0; i < A.rows(); ++i) 
  {
    for (int j = 0; j < A.colms(); ++j) 
    {
      printf("%13.5f", A(i, j));
    }
    puts("");
  }
  puts("");
}

template<bool T, bool O>
void populate_matrix(MatrixT<T, O> &A)
{  
  static std::minstd_rand rng{9999};
  static std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (int i = 0; i < A.rows(); ++i) 
  {
    for (int j = 0; j < A.colms(); ++j) 
    {
      A(i, j) = dist (rng);
    }
  }
}

template <bool Asc, bool Row, bool T1, bool O1, bool T2, bool O2>
MatrixPair sorts(const MatrixT<T1, O1> &Diag, MatrixT<T2, O2> &Orth)
{
  const int m = Diag.rows();
  const int s = Orth.rows();
  const int t = Orth.colms();
  Matrix D (m, m);
  Matrix O (s, t);
  std::vector<int> idx (m);
  std::iota (idx.begin(), idx.end(), 0);
  
  sort(idx.begin(), idx.end(), [&](int i, int j)
  {
    if constexpr (Asc)
    {
      return Diag(i, i) < Diag(j, j);
    }
    else 
    {
      return Diag(i, i) > Diag(j, j);
    }
  });

  if constexpr (Row)
  {
    for (int i = 0; i < s; ++i)
    {
      const int j = idx[i];
      D (i, i) = Diag (j, j);
      Matrix::row_immerse( Orth.row_extract(j), O, i);
    } 
  }
  else
  {
    for (int i = 0; i < t; ++i)
    {
      const int j = idx[i];
      D (i, i) = Diag (j, j);
      Matrix::column_immerse( Orth.column_extract(j), O, i);
    } 
  }

  return MatrixPair(D, O);
}

template<bool T, bool O>
Correction block_diagonal(MatrixT<T, O> &Sym)
{
  const int n = Sym.rows();
  Matrix Cor (n, 1);

  const double mid = n / 2;
  const double rho = Sym(mid, mid - 1);
  
  Cor(mid , 0) = Cor(mid - 1, 0) = 1;

  Sym(mid, mid - 1) = Sym(mid - 1, mid) = 0;
  Sym(mid, mid) -= rho;
  Sym(mid - 1, mid - 1) -= rho;

  return std::make_pair(rho, Cor);
}

template<bool T, bool O>
void gram_schmidt(MatrixT<T, O> &U, const int i)
{
  const int n = U.colms();
  for (int j = i; j < n; ++j)
  {

    Matrix res (n, 1);
    //random column vector
    Matrix r (n, 1);
    populate_matrix(r);

    #pragma omp parallel for schedule (auto) reduction (mat_add : res)
    for ( int k = 0; k < i; ++k)
    {
      Matrix extracted = U.column_extract(k);
      const double numer = (extracted.transpose() * r)(0, 0);
      const double denom = (extracted.transpose() * extracted)(0, 0);
      
      res = res + (numer * extracted) * (1 / denom);
    }

    U = Matrix::column_immerse(r - res, U, j);
  }
}

template<bool T, bool O>
Matrix s_construction(const MatrixT<T, O> &Init, const MatrixT<T, O> &Eig)
{
  const int n = Init.rows();
  const int m = Init.colms();
  Matrix S (n, m);

  #pragma omp parallel for
  for (int i = 0; i < n; ++i)
  {
    S(i, i) = std::sqrt(Eig(i, i));
  }

  return S;
}

template<bool T1, bool O1, bool T2, bool O2, bool T3, bool O3>
Matrix u_construction(const MatrixT<T1, O1> &Init, MatrixT<T2, O2> &Orth, const MatrixT<T3, O3> &S)
{
  const int n = Init.rows(); 
  const int m = Init.colms(); // Dimensions of ortho are m x m

  Matrix U (n, n);
  
  #pragma omp parallel for
  for (int i = 0; i < m; ++i)
  {
    const double s = S(i, i);
    Matrix C = (1 / s) * Init * Orth.column_extract(i);
    U = Matrix::column_immerse(C, U, i);
  }

  gram_schmidt(U, m);
  
  return U;
}

template<bool T, bool O>
Matrix initial_e_approx(const MatrixT<T, O> &Diag, const Correction &Cor)
{
  const int m = Diag.rows();
  const auto & [rho, Z] = Cor;
  
  //Matrix of initial approximations
  Matrix Y (m, m);

  for (int k = 0; k < m - 1; ++k)
  {
    const double delta = Diag(k + 1, k + 1) - Diag(k, k);
    const double mid = (Diag(k + 1, k + 1) + Diag(k, k)) / 2;
    const double w = secular_function(Diag, Z, rho, mid);
    const double c = g_function(Diag, Z, rho, mid, k);
    const double z = Z(k, 0);
    const double z_next = Z(k + 1, 0);
    double a = 0;
    double b = 0;
    //buffer counter
    int K = 0;

    if (w >= 0)
    {
      K = k;
      a = (c * delta) + ((z * z) + (z_next * z_next));
      b = z * z * delta;
    }

    else 
    {
      K = ++k;
      a = -(c * delta) + ((z * z) + (z_next * z_next));
      b = -z_next * z_next * delta;
    }

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));

    if (a <= 0)
    {
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(K, K) + ((a - root) / (2 * c));
    }

    else 
    {
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(K, K) + ((2 * b) / (a + root));
    }
  }


  //edge case k = m - 1
  const int k = m - 1;
  //finding the upper limit for eigenvalues:
  const double d_max = Diag(k, k) + ((Z.transpose() * Z)(0, 0) / rho);
  const double mid = (d_max + Diag(k, k)) / 2;
  const double g = g_function(Diag, Z, rho, mid, k);
  const double h = h_function(Diag, Z, d_max, k);//
  const double w = secular_function(Diag, Z, rho, mid);
  cout << "f: " << w << "\n";
  if ((w <= 0) && (g <= -h))
  {
    cout << "FIRST\n";
    Y(k, k) = d_max;
  }

  else
  {
    cout << "SECOND\n";
    const double c = g;
    const double delta = Diag(k, k) - Diag(k - 1, k - 1);
    const double z = Z(k, 0);
    const double z_prev = Z(k - 1, 0);
    const double a = -(c * delta) + ((z * z) + (z_prev * z_prev));
    const double b = -z * z * delta;

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));

    if (a >= 0)
    {
      cout << "1\n";
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(k, k) + ((a + root) / (2 * c));
    }
    else 
    {
      cout << "2\n";
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(k, k) + ((2 * b) / (a - root));
    }
  }
  return Y;
}

/************************************************************/
// Major function prototypes

template <bool T, bool O>
void tridiagonalization(MatrixT<T, O> &Sym)
{
  // producing a column vector, replicating the last column of Sym
  double alpha, RSQ;
  const int n = Sym.rows();
  
  for (int k = 0; k < n - 2; ++k) {

    alpha = RSQ = 0;
    Matrix W(Sym.rows(), 1);

    // for k = 0 ...  < n-2
    for (int i = k + 1; i < n; ++i) 
    {
      alpha += std::pow(Sym(i, k), 2);
      W(i, 0) = (0.5) * Sym(i, k);
    }

    const double leadingVal = Sym(k + 1, k);
    //final alpha definition
    alpha = -(std::sqrt(alpha)); 
    //represents 2r^2
    RSQ = std::pow(alpha, 2) - (alpha * leadingVal); 
    
    //leading entry in w-vector
    W(k + 1, 0) = (0.5) * (leadingVal - alpha); 

    auto WTW = W * W.transpose(); 
    WTW = Matrix::identity(n) + ((-4 / RSQ) * WTW);
    //transforming the original matrix
    Sym = WTW * Sym * WTW;
  }
}

template <bool T, bool O>
Matrix secular_solver(const MatrixT<T, O> &Diag, const Correction &Cor)
{
  const double accuracy = std::pow(10, -8);
  const int m = Diag.rows();
  const auto & [rho, Z] = Cor;

  //defining a matrix with initial eigenvalue approximations:
  auto Y = initial_e_approx(Diag, Cor);

  cout << "Initial: \n" ;
  print_matrix(Y);
  //finding the upper limit for eigenvalues:
  
  for (int k = 0; k < m - 1; ++k)
  {
    double approx = 1;
    while(std::abs(approx) > accuracy)
    {
      //small necessary computations
      const double y = Y(k, k);
      const double delta = Diag(k, k) - y;
      const double delta_next = Diag(k + 1, k + 1) - y;
      const double w = secular_function(Diag, Z, rho, y);
      const double w_ = secular_function_prime(Diag, Z, y);
      const double psi_ = psi_prime(Diag, Z, y, k);
      const double phi_ = phi_prime(Diag, Z, y, k);

      //saving a computation
      const double buf = delta * delta_next;

      const double a = ((delta + delta_next) * w) - (buf * w_);
      const double b = buf * w;
      const double c = w - (delta * psi_) - (delta_next * phi_);

      //saving a computation
      const double root = std::sqrt((a * a) - (4 * b * c));

      if (a <= 0)
      {
        //updating the approximation matrix
        approx = (a - root) / (2 * c);
      }

      else
      {
        //updating the approximation matrix
        approx= (2 * b) / (a + root);
      }
      Y(k, k) += approx;
    }
  }

  while (1)
  {
    //edge case k = m - 1
    const int k = m - 1;
    const double y = Y(k, k);
    const double delta = Diag(k, k) - y;
    const double delta_prev = Diag(k - 1, k - 1) - y;
    const double w = secular_function(Diag, Z, rho, y);
    const double w_ = secular_function_prime(Diag, Z, y);
    const double psi_ = psi_prime(Diag, Z, y, k - 1);

    //saving a computation
    const double buf = delta * delta_prev;

    const double a = ((delta + delta_prev) * w) - (buf * w_);
    const double b = buf * w;
    const double c = w - (delta_prev * psi_) - (Z(k, 0) * Z(k, 0) / delta);

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));
    double approx = 0;

    if (a >= 0)
    {
      //updating the approximation matrix
      approx = (a + root) / (2 * c);
    }

    else
    {
      //updating the approximation matrix
      approx = (2 * b) / (a - root);
    }

    if (approx >= accuracy)
    {
      Y(k, k) += accuracy;
    }

    else
    {
      break;
    }
  }
  return Y;  
}

template <bool T1, bool O1, bool T2, bool O2>
Matrix evector_extract(const MatrixT<T1, O1> &Eig, const MatrixT<T2, O2> &Diag)
{
  const int m = Eig.rows();
  Matrix Evec (m, m);
  Matrix Z (m, 1);

  //computing approximation to each z
  //#pragma omp parallel
  for(int i = 0; i < m; ++i)
  {
    double product1 = 1;
    double product2 = 1;

    if(!(i == 0))
    {
      //#pragma omp for reduction(* : product1)
      for (int j = 0; j < i; ++j)
      {
        product1 *= (Eig(j, j) - Diag(i, i)) / (Diag(j, j) - Diag(i, i)); 
      }
    }

    if(!(i == (m - 1)))
    {
      //#pragma omp for reduction(* : product2)
      for (int k = i; k < (m - 1); ++k)
      {
        product2 *= (Eig(k, k) - Diag(i, i)) / (Diag(k + 1, k + 1) - Diag(i, i));
      }
    }

    double product3 = Eig(m - 1, m - 1) - Diag(i, i);

    Z(i, 0) = std::sqrt(product1 *product2 * product3);
    
  }

  print_matrix(Z);


  Matrix Q (m, 1);
  //computing approximation to each eigenvector
  //#pragma omp parallel
  for(int i = 0; i < m; ++i)
  {
    //#pragma omp for
    for(int j = 0; j < m; ++j)
    {
      Q(j, 0) = Z(j, 0) / (Diag(j, j) - Eig(i, i)); 
    }
  
    double sum = 0;
    //#pragma omp for reduction(+ : sum)
    for(int k = 0; k < m; ++k)
    {
      double term = Z(k, 0) / (Diag(k, k) - Eig(i, i));
      sum += term * term;
    }
    
    Q = Q * (1 / std::sqrt(sum));
    
    Matrix::column_immerse(Q, Evec, i);
  }
  
  return Evec;
}

template <bool T, bool O>
MatrixPair par_eigen_decomp(MatrixT<T, O> &Sym, unsigned dep)
{
  const int n = Sym.rows();
  
  if (n == 1)
  {
    Matrix Orth (n, n);
    Orth(0,0) = 1;

    return MatrixPair(Orth, Sym);
  }

  else if (n == 2) 
  {
    print_matrix(Sym);
    const double a  = Sym(0, 0);
    const double d = Sym(1, 1);
    const double c  = Sym(1, 0);
    
    Matrix Orth (n, n);
    Matrix Diag  (n, n);

    const double v = sqrt((a + d) * (a + d) - (4 * (a * d - c * c)));
    const double l1 = ((a + d) + v) / 2; 
    const double l2 = ((a + d) - v) / 2;
    
    Diag(0, 0) = l1;
    Diag(1, 1) = l2;

    //eigenvector magnitudes
    const double v12 = ((l1 - a) / c);
    const double v22 = ((l2 - a) / c);
    const double v1m = sqrt( 1 + pow( v12, 2));
    const double v2m = sqrt( 1 + pow( v22, 2));

    Orth(0, 0) = 1.0 / v1m;
    Orth(0, 1) = 1.0 / v2m;
    Orth(1, 0) = v12 / v1m;
    Orth(1, 1) = v22 / v2m;

    print_matrix(Diag);

    return MatrixPair(Orth, Diag);
  } 
  else 
  {
    if ( dep == 0)
    {
      return eigen_decomp(Sym);
    }
    else 
    {
      Correction Cor = block_diagonal(Sym);
      
      auto Hi = Sym.cut( n / 2, 1);
      auto Lo = Sym.cut(n / 2, 0); 

      --(--dep);

      std:: future t1 = std::async (std::launch::async, par_eigen_decomp<T, O>, std::ref(Hi), dep);
      std:: future t2 = std::async (std::launch::async, par_eigen_decomp<T, O>, std::ref(Lo), dep);

      const auto & [Orth1, Diag1] = t1.get();
      const auto & [Orth2, Diag2] = t2.get();
    
      const auto Orth  = Matrix::combine (Orth1, Orth2);
      auto Diag   = Matrix::combine (Diag1, Diag2);
      
      const auto OrthT = Orth.transpose();
      const auto & [scalar, unitVector] = Cor;
      auto Z = (1 / (sqrt(2))) * (OrthT * unitVector);
      double rho = 1 / (2 * scalar);
      
      if (rho < 0)
      {
        rho = -rho;
        Z = -1 * Z;
        Diag = -1 * Diag;

        auto [D, U] = sorts<true, true>(Diag, Z); 
        Cor = std::make_pair(rho, U);
        auto Eval = secular_solver(D, Cor);
        auto Evec = evector_extract(Eval, D);

        Eval = -1 * Eval;
        Evec = -1 * Evec;

        auto [Eva, Eve] = sorts<false, false>(Eval, Evec); 
        return MatrixPair(Eve, Eva);
      }
      else
      {
        auto [D, U] = sorts<true, true>(Diag, Z); 
        Cor = std::make_pair(rho, U);
        auto Eval = secular_solver(D, Cor);
        auto Evec = evector_extract(Eval, D);
        const auto [Eva, Eve] = sorts<false, false>(Eval, Evec); 
          
        return MatrixPair(Eve, Eva);
      }
    }
  }
}

template <bool T, bool O>
MatrixPair eigen_decomp(MatrixT<T, O> &Sym)
{
  const int n = Sym.rows();
  
  if (n == 1)
  {
    Matrix Orth (n, n);
    Orth(0,0) = 1;

    return MatrixPair(Orth, Sym);
  }

  else if (n == 2) 
  {
    print_matrix(Sym);
    const double a  = Sym(0, 0);
    const double d = Sym(1, 1);
    const double c  = Sym(1, 0);
    
    Matrix Orth (n, n);
    Matrix Diag  (n, n);

    const double v = sqrt((a + d) * (a + d) - (4 * (a * d - c * c)));
    const double l1 = ((a + d) + v) / 2; 
    const double l2 = ((a + d) - v) / 2;
    
    Diag(0, 0) = l1;
    Diag(1, 1) = l2;

    //eigenvector magnitudes
    const double v12 = ((l1 - a) / c);
    const double v22 = ((l2 - a) / c);
    const double v1m = sqrt( 1 + pow( v12, 2));
    const double v2m = sqrt( 1 + pow( v22, 2));

    Orth(0, 0) = 1.0 / v1m;
    Orth(0, 1) = 1.0 / v2m;
    Orth(1, 0) = v12 / v1m;
    Orth(1, 1) = v22 / v2m;

    print_matrix(Diag);

    return MatrixPair(Orth, Diag);
  } 
  else 
  {
    Correction Cor = block_diagonal(Sym);
    
    auto Hi = Sym.cut( n / 2, 1);
    auto Lo = Sym.cut(n / 2, 0); 

    const auto & [Orth1, Diag1] = eigen_decomp(Hi);
    const auto & [Orth2, Diag2] = eigen_decomp(Lo);

    const auto Orth  = Matrix::combine (Orth1, Orth2); //needed solely for testing purposes
    auto Diag   = Matrix::combine (Diag1, Diag2);

    
    const auto OrthT = Orth.transpose();
    const auto & [scalar, unitVector] = Cor;
    auto Z = (1 / (sqrt(2))) * (OrthT * unitVector);
    double rho = 1 / (2 * scalar);


    /*
    //check for the decomposition correctness
    auto corr = (1 / rho) * ( Z * Z.transpose());
    auto sec = Diag + corr;
    auto thir = Orth * sec * OrthT;

    cout <<"This has to be equal the original\n";
    print_matrix(thir);
    */
    
    if (rho < 0)
    {
      rho = -rho;
      Z = -1 * Z;
      Diag = -1 * Diag;

      auto [D, U] = sorts<true, true>(Diag, Z); 
      Cor = std::make_pair(rho, U);
      auto Eval = secular_solver(D, Cor);
      auto Evec = evector_extract(Eval, D);

      Eval = -1 * Eval;
      Evec = -1 * Evec;

      auto [Eva, Eve] = sorts<false, false>(Eval, Evec); 
      return MatrixPair(Eve, Eva);
    }
    else
    {
      auto [D, U] = sorts<true, true>(Diag, Z); 
      Cor = std::make_pair(rho, U);

      cout << "Corr: \n" ;
      print_matrix(U);
      cout << "Diag: \n" ;
      print_matrix(D);
      
      // Retrieving eigenvalues from secular equation.
      auto Eval = secular_solver(D, Cor);
      cout << "Eigenvalues:\n";
      print_matrix(Eval);
      // Calculating eigenvectors from defined eigenvalues.
      auto Evec = evector_extract(Eval, D);
      cout << "Eigenvector:\n";
      print_matrix(Evec);
      // Fixing order of diagonal and orthogonal entries.
      const auto [Eva, Eve] = sorts<false, false>(Eval, Evec); 
        
      return MatrixPair(Eve, Eva);
    }
  }
}

template <bool T, bool O>
MatrixTuple singular_value_decomp(const MatrixT<T, O> &Init)
{
  auto Sym =  Init.transpose() * Init;
  //tridiagonalization(Sym);
  auto [Eve, Eva] = eigen_decomp(Sym);

  auto S = s_construction(Init, Eva);
  auto U = u_construction(Init, Eve, S);

  print_matrix(U);
  print_matrix(S);
  print_matrix(Eve.transpose());

  return std::make_tuple(U, S, Eve.transpose());
}

#endif