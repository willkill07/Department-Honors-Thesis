/*
  Author     : Maksim Y. Melnichenko
  Title      : SVD Inplementation

  The following project is intended to design an implementation 
  Singular Value Decomposition (Ref.4) feature extraction algorithm. SVD is a special technique of real matrix factorization, 
  which, unlike Eigen decomposition, could be applied to rectangular matrices, extracting special “singular values.” SVD, 
  whenever discussed in an applied sense, could be discussed in a number of different ways. Considering the properties of matrices 
  that are being produced after the application of SVD, I would define the essence of SVD method as such: it is a method that allows 
  us to identify and order dimensions, along which data points exhibit the most variation. Once we assess this info, we may 
  identify the best approximation for the original data points using fewer dimensions. 
  Therefore, SVD provides a method for data reduction. 

  The routines, provided so far, represent the necessary steps for the initial USV^T-factorization of the original matrix.
  Be advised that this is a work in progress. 
  All algorithms are first being overviewed on paper and are then represented as C++ code. 
  The routines are designed with hope for future optimization and parallelization.

  References:
  1 - Numerical Analysis (Richard L Burden; J Douglas Faires; Annette M Burden) ISBN-13: 978-1305253667
  2 - J. J. M. Cuppen, A divide and conquer method for the symmetric tridiagonal eigenproblem, Numer. Math., 36 (1981), pp. 177–195. 
  3 - http://www.netlib.org/lapack/lawnspdf/lawn89.pdf
  4 - https://en.wikipedia.org/wiki/Singular_value_decomposition
  

/************************************************************/
// System includes

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

//custom matrix header
#include "Matrix.h"
/************************************************************/
// Using declarations

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
// Major function prototypes

void trigiagonalization(Matrix &Sym);

MatrixPair eigen_decomp(Matrix &Sym);

Matrix secular_solver(const Matrix &Diag, const Correction &Cor);

Matrix evector_extract(const Matrix &Eig, const Matrix &Diag);

MatrixTuple singular_value_decomp(const Matrix &Init, const Matrix &Eig, const Matrix &Orth);

/************************************************************/
// Helper function prototypes

void populate_matrix(Matrix &Init);

Correction block_diagonal(Matrix &Sym);

Matrix initial_e_approx(const Matrix &Diag, const Correction &Cor);

Matrix s_construction(const Matrix &Init, const Matrix &Eig);

Matrix u_construction(const Matrix &Init, const Matrix &Orth, const Matrix &S);

void gram_schmidt(Matrix &U, int i);

template <bool O, bool T>
void print_matrix(const MatrixT<O,T> &A);

/************************************************************/
// Small mathematical functions

double secular_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double secular_function_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double psi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double phi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double g_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double h_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

/************************************************************/
/*
 Main Function - At this stage, is populated with the testing code and matrix examples to be tested.
 In order to observe the testing results, one shall use the print_matrix function.
 */ 

int main(int argc, char *argv[]) {

  /*
  //
  Building an initial Matrix and populating it with randomly-generated values.
  //

  int r = 10;
  int c = 15;

  Matrix A (r, c);
  populate_matrix(A);
  auto B = A.transpose() * A;

  //
  During the testing phases of the project, the manually-typed example matrices were being used.
  Such matrix, initially present in an example from (Ref.1 page 606), may be viewed below.
  //
  */
  //TEST Tridiag - book
  Matrix TEST_B(4, 4);

  TEST_B(0, 0) = 4;
  TEST_B(0, 1) = 1;
  TEST_B(0, 2) = -2;
  TEST_B(0, 3) = 2;

  TEST_B(1, 0) = 1;
  TEST_B(1, 1) = 2;
  TEST_B(1, 2) = 0;
  TEST_B(1, 3) = 1;

  TEST_B(2, 0) = -2;
  TEST_B(2, 1) = 0;
  TEST_B(2, 2) = 3;
  TEST_B(2, 3) = -2;

  TEST_B(3, 0) = 2;
  TEST_B(3, 1) = 1;
  TEST_B(3, 2) = -2;
  TEST_B(3, 3) = -1;


  //TEST DNC - Large
  Matrix TEST_L (4, 4);

  TEST_L(0, 0) = 7;
  TEST_L(0, 1) = 3;
  TEST_L(0, 2) = 0;
  TEST_L(0, 3) = 0;

  TEST_L(1, 0) = 3;
  TEST_L(1, 1) = 1;
  TEST_L(1, 2) = 2;
  TEST_L(1, 3) = 0;

  TEST_L(2, 0) = 0;
  TEST_L(2, 1) = 2;
  TEST_L(2, 2) = 8;
  TEST_L(2, 3) = -2;

  TEST_L(3, 0) = 0;
  TEST_L(3, 1) = 0;
  TEST_L(3, 2) = -2;
  TEST_L(3, 3) = 3;
  
  //TEST DNC - Small
  Matrix TEST_S (2, 2);

  TEST_S(0, 0) = 1;
  TEST_S(0, 1) = 3;
  TEST_S(1, 0) = 3;
  TEST_S(1, 1) = 2;


  trigiagonalization(TEST_S);
  auto [ortho, eigenvalues] = eigen_decomp(TEST_S);
  

}


/*
trigiagonalization - Implementation of Householder's method (Ref. 1 page 602) of turning a
symmetric matrix into a symmetric tridiagonal matrix. Modifies the original by refernce,
uses a helper routine for the similarity transformation matrix production.
*/


void trigiagonalization(Matrix &Sym) {

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


/*
dNc_cuppen - Cuppen's Divide and Conquer eigenvalue extraction algorithm (Ref. 2) - a recursive
algorithm, consisting of two parts. The main intention of the "Divide" portion
is to formulate a secular equation, represented by finite series, 
the roots of which will provide the eigenvalues of the original matrix.
The "Conquer" protion involves solving the secular equation.
So far, the smallest undividible has been set to be represented by a 2x2 matrix, which means that
the algorithm will only provide correct results for even-dimensional matrices. This problem will
be updated, as the appropriate solution technique for the seqular equation will be derived.
*/
MatrixPair eigen_decomp(Matrix &Sym)
{
  int n = Sym.rows();

  if (n == 1)
  {
    Matrix Orth (n, n);
    Orth(0,0) = 1;

    return MatrixPair(Orth, Sym);
  }

  else if (n == 2) 
  {

    const double a  = Sym(0, 0);
    const double d = Sym(1, 1);
    const double c  = Sym(1, 0);
    
    Matrix Orth (n, n);
    Matrix Diag  (n, n);

    const double v = sqrt((a + d) * (a + d) - (4 * (a * d - c * c)));
    const double l1 = ((a + d) + v) / 2;; 
    const double l2 = ((a + d) + v) / 2;;
    
    Diag(0, 0) = l1;
    Diag(1, 1) = l2;

    //eigenvector magnitudes
    double v12 = ((l1 - a) / c);
    double v22 = ((l2 - a) / c);
    double v1m = sqrt( 1 + pow( v12, 2));
    double v2m = sqrt( 1 + pow( v22, 2));

    Orth(0, 0) = 1.0 / v1m;
    Orth(0, 1) = 1.0 / v2m;
    Orth(1, 0) = v12 / v1m;
    Orth(1, 1) = v22 / v2m;
    
    return MatrixPair(Orth, Diag);
  } 
  else 
  {
    Correction Cor = block_diagonal(Sym);

    Matrix Hi = Sym.cut( n / 2, 1);
    Matrix Lo = Sym.cut(n - (n / 2), 0);    

    const auto & [Orth1, Diag1] = eigen_decomp(Hi);
    const auto & [Orth2, Diag2] = eigen_decomp(Lo);

    Matrix Orth  = Matrix::combine (Orth1, Orth2);
    auto OrthT = Orth.transpose();
    Matrix Diag   = Matrix::combine (Diag1, Diag2);

    const auto & [scalar, unitVector] = Cor;

    Matrix Z = (1 / (sqrt(2))) * (OrthT * unitVector);
    Cor = std::make_pair(2 * scalar, Z);

    /* check for the decomposition correctness
    auto corr = Cor.first * ( Cor.second * Cor.second.transpose());
    auto sec = diag + corr;
    auto thir = ortho * sec * orthoT;

    cout <<"This has to be equal the original\n";
    print_matrix(thir);
    */

    // Fixing order of diagonal and orthogonal entries.
    Matrix::sorts<0>(Diag, Orth); 
    // Retrieving eigenvalues from secular equation.
    Matrix Evalue = secular_solver(Diag, Cor);
    // Calculating eigenvectors from defined eigenvalues.
    Matrix Evector = evector_extract(Evalue, Diag);
    // Fixing order of diagonal and orthogonal entries.
    Matrix::sorts<1>(Evalue, Evector);
    
    
    return MatrixPair (Evector, Evalue);
  }
}


Matrix secular_solver(const Matrix &Diag, const Correction &Cor, Matrix)
{
  const int m = Diag.rows();
  const auto & [rho, Z] = Cor;

  //defining a matrix with initial eigenvalue approximations:
  Matrix Y = initial_e_approx(Diag, Cor);

  //finding the upper limit for eigenvalues:
  const double d_max = Diag(m, m) + (rho * (Z.transpose() * Z)(0, 0));

  for (int k = 0; k < m - 1; ++k)
  {
    //small necessary computations
    const double y = Y(k, k);
    const double delta = Diag(k, k) - y;
    const double delta_next = Diag(k + 1, k + 1) - y;
    const double w = secular_function(Diag, Z, rho, y);
    const double w_ = secular_function_prime(Diag, Z, rho, y);
    const double psi_ = psi_prime(Diag, Z, rho, y, k);
    const double phi_ = phi_prime(Diag, Z, rho, y, k);

    //saving a computation
    const double buf = delta * delta_next;

    const double a = (delta + delta_next) * w - (buf * w_);
    const double b = buf * w;
    const double c = w - (delta * psi_) - (delta_next * phi_);

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));

    if (a <= 0)
    {
      //updating the approximation matrix
      Y(k, k) += (a - root) / (2 * c);
    }

    else
    {
      //updating the approximation matrix
      Y(k, k) += (2 * b) / (a + root);
    }
  }

  //edge case k = m - 1
  const int k = m - 1;
  const double y = Y(m - 1, m - 1);
  const double delta = d_max - y;
  const double delta_prev = Diag(k, k) - y;
  const double w = secular_function(Diag, Z, rho, y);
  const double w_ = secular_function_prime(Diag, Z, rho, y);
  const double psi_ = psi_prime(Diag, Z, rho, y, k);
  const double phi_ = phi_prime(Diag, Z, rho, y, k);

  //saving a computation
  const double buf = delta * delta_prev;

  const double a = (delta + delta_prev) * w - (buf * w_);
  const double b = buf * w;
  const double c = w - (delta_prev * psi_) - (Z(k, k) * Z(k, k) / delta);

  //saving a computation
  const double root = std::sqrt((a * a) - (4 * b * c));

  if (a >= 0)
  {
    //updating the approximation matrix
    Y(k, k) += (a - root) / (2 * c);
  }

  else
  {
    //updating the approximation matrix
    Y(k, k) += (2 * b) / (a + root);
  }


  return Y;  
}

/*
Matrix evector_extract(const Matrix &Eig, const Matrix &Diag)
{
  const int m = Eig.rows();
  Matrix Evec (m, m);
  Matrix Z (m, 1);

  //computing approximation to each z
  #pragma omp parallel
  for(int i = 0; i < m; ++i)
  {
    auto product1 = 0;
    #pragma omp for reduction(* : product1)
    for (int j = 0; j < (i - 1); ++j)
    {
      product1 *= (Eig(j, j) - Diag(i, i)) / (Diag(j, j) - Diag(i, i));
    }

    auto product2 = 0;
    #pragma omp for reduction(* : product2)
    for (int k = 0; k < (m - 1); ++k)
    {
      product2 *= (Eig(k, k) - Diag(i, i)) / (Diag(k + 1, k + 1) - Diag(i, i));
    }

    auto product3 = Eig(m, m) - Diag(i, i);

    Z(i, 1) = std::sqrt(product1 *product2 * product3);
  }

  //computing approximation to each eigenvector
  #pragma omp parallel
  for(int i = 0; i < m; ++i)
  {
    #pragma omp for
    for(int j = 0; j < m; ++j)
    {
      Z(j, 1) = Z(j, 1) / (Diag(j, j) - Eig(i, i)); 
    }

    auto sum = 0;
    #pragma omp for reduction(+ : sum)
    for(int k = 0; k < m; ++k)
    {
      auto term = Z(k, k) / (Diag(k, k) - Eig(i, i));
      sum += term * term;
    }
    Z = Z * (1 / sum);

    Matrix::column_immerse(Z, Evec, i);
  }

  return Evec;
}
*/
MatrixTuple singular_value_decomp(const Matrix &Init, const Matrix &Eig, const Matrix &Orth)
{
  Matrix S = s_construction(Init, Eig);
  Matrix U = u_construction(Init, Orth, S);
  return std::make_tuple(U, S, Orth.transpose());
}

/*
populate_matrix - by reference, takes in the orignal matrix 
and populates it by assigning a random floating point number to each of its cells.
*/
void populate_matrix(Matrix &A) {
  
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

/*
block_diagonal - routine that makes the original matrix, taken in by reference, 
block diagonal and additionally updates the "factored-out" matrix Cor with corresponding
elements. Serves as a helper subroutine for Cuppen's DnC algorithm.
*/

Correction
block_diagonal(Matrix &Sym)
{
  int n = Sym.rows();
  Matrix Cor (n, 1);

  double mid = n / 2;
  double rho = Sym(mid, mid - 1);
  
  Cor(mid , 0) = Cor(mid - 1, 0) = 1;

  Sym(mid, mid - 1) = Sym(mid - 1, mid) = 0;
  Sym(mid, mid) -= rho;
  Sym(mid - 1, mid - 1) -= rho;

  return std::make_pair(rho, Cor);
}



Matrix initial_e_approx(const Matrix &Diag, const Correction &Cor)
{
  const int m = Diag.rows();
  const auto & [rho, Z] = Cor;
  //finding the upper limit for eigenvalues:
  const double d_max = Diag(m, m) + (rho * (Z.transpose() * Z)(0, 0));
  //Matrix of initial approximations
  Matrix Y (m, m);

  for (int k = 0; k < m - 1; ++k)
  {
    const double delta = Diag(k + 1, k + 1) - Diag(k, k);
    const double mid = (Diag(k + 1, k + 1) + Diag(k, k)) / 2;
    const double f = secular_function(Diag, Z, rho, mid);
    double a = 0;
    double b = 0;
    const double c = g_function(Diag, Z, rho, mid);
    //buffer counter
    int K = 0;

    if (f >= 0)
    {
      K = k;
      const double z = Z(k, k);
      const double z_next = Z(k + 1, k + 1);

      a = c * delta + ((z * z) + (z_next * z_next));
      b = z * z * delta;
    }

    else 
    {
      K = k + 1;
      const double z = Z(k, k);
      const double z_next = Z(k + 1, k + 1);

      a = -c * delta + ((z * z) + (z_next * z_next));
      b = -z * z * delta;
    }

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));

    if (a <= 0)
    {
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(k, K) + (a - root) / (2 * c);
    }

    else 
    {
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(k, K) + (2 * b) / (a + root);
    }
  }


  //edge case k = m - 1
  const int k = m - 1;
  const double mid = (d_max + Diag(k, k)) / 2;
  const double g = g_function(Diag, Z, rho, mid);
  const double h = h_function(Diag, Z, rho, d_max);

  if (g <= -h)
  {
    Y(k, k) = Diag(k, k) + (rho * (Z.transpose() * Z)(0, 0));
  }

  else
  {
    const double c = g;
    const double delta = Diag(k, k) - Diag(k - 1, k - 1);
    const double z = Z(k, k);
    const double z_prev = Z(k - 1, k - 1);
    const double a = -c * delta + ((z * z) + (z_prev * z_prev));
    const double b = -z * z * delta;

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));

    if (a >= 0)
    {
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(k, k) + (a - root) / (2 * c);
    }

    else 
    {
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(k, k) + (2 * b) / (a + root);
    }

  }

  return Y;
}

/*
s_construction - construction a matrix of singular values with dimensions,
identical to the original matrices'.
Precondition: eigenvalue diagonal matrix, sorted in descending order;
known dimensions of the original matrix.
*/
Matrix s_construction(const Matrix &Init, const Matrix &Eig)
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


/*
The V^T matirx comes from the transposition of an orthogonal matrix.
*/


/* build a fuction that extracts/ puts vectors back in
Matrix u_construction(const Matrix &Init, const Matrix &Orth, const Matrix &S)
{
  const int n = Init.rows(); 
  const int m = Init.colms(); // Dimensions of ortho are m x m

  Matrix U (n, n);

  #pragma omp parallel for
  for (int i = 0; i < m; ++i)
  {
    double s = S(i, i);
    Matrix C = (1 / s) * Init * Matrix::column_extract(Orth, i);
    U = Matrix::column_immerse(C, U, i);
  }

  gram_schmidt(U, m);

  return U;
}

void gram_schmidt(Matrix &U, int i)
{
  const int n = U.rows();
  
  #pragma omp parallel
  for (int j = i; j < n; ++j)
  {

    Matrix res (n, 1);
    //random column vector
    Matrix r (n, 1);
    populate_matrix(r);

    #pragma omp for schedule (auto) reduction (- : res)
    for ( int k = 0; k < (n - i); ++k)
    {
      Matrix extracted = Matrix::column_extract(U, k);
      double magnitude = Matrix::magnitude(extracted);
      res += ((extracted * r) * extracted) * (1 / (magnitude * magnitude));
    }

    U = Matrix::column_immerse(r - res, U, j);
  }
}

/*
Printing - Simple routine, created for the testing purposes.
*/
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


double secular_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y)
{

}

double secular_function_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y)
{

}

double psi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
{

}

double phi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
{

}

double g_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y)
{

}

double h_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y)
{
  
}