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
  */

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
#include "Functions.h"
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

void tridiagonalization(Matrix &Sym); //

MatrixPair eigen_decomp(Matrix &Sym); //

Matrix secular_solver(Matrix &Diag, const Correction &Cor);

Matrix evector_extract(const Matrix &Eig, const Matrix &Diag);

MatrixTuple singular_value_decomp(const Matrix &Init);

/************************************************************/
// Helper function prototypes

void populate_matrix(Matrix &Init); //

Correction block_diagonal(Matrix &Sym); //

Matrix initial_e_approx(const Matrix &Diag, const Correction &Cor);

Matrix s_construction(const Matrix &Init, const Matrix &Eig); //

Matrix u_construction(const Matrix &Init,  Matrix &Orth, const Matrix &S); //

void gram_schmidt(Matrix &U, int i); //

template <bool Asc, bool Row, bool T1, bool O1, bool T2, bool O2>
MatrixPair sorts(const MatrixT<T2, O2> &Diag, MatrixT<T2, O2> &Orth);

template <bool O, bool T>
void print_matrix(const MatrixT<O,T> &A); //

/************************************************************/
// Small mathematical functions

double secular_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double secular_function_prime(const Matrix& Diag, const Matrix& Z, const double y);

double psi(const Matrix& Diag, const Matrix& Z, const double y, const int k);

double phi(const Matrix& Diag, const Matrix& Z, const double y, const int k);

double psi_prime(const Matrix& Diag, const Matrix& Z, const double y, const int k);

double phi_prime(const Matrix& Diag, const Matrix& Z, const double y, const int k);

double g_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double h_function(const Matrix& Diag, const Matrix& Z, const double y, const int k);







Matrix initial_e_approx_negative(const Matrix &Diag, const Correction &Cor);


/************************************************************/
/*
 Main Function - At this stage, is populated with the testing code and matrix examples to be tested.
 In order to observe the testing results, one shall use the print_matrix function.
 */ 

int main() {

  /*
  //
  Building an initial Matrix and populating it with randomly-generated values.
  //
  omp_set_numthreads(10);

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

  Matrix TEST_C(4, 4);

  TEST_C(0, 0) = 6;
  TEST_C(0, 1) = 1;
  TEST_C(0, 2) = 0;
  TEST_C(0, 3) = 0;

  TEST_C(1, 0) = 1;
  TEST_C(1, 1) = 4;
  TEST_C(1, 2) = 5;
  TEST_C(1, 3) = 0;

  TEST_C(2, 0) = 0;
  TEST_C(2, 1) = 5;
  TEST_C(2, 2) = 2;
  TEST_C(2, 3) = 3;

  TEST_C(3, 0) = 0;
  TEST_C(3, 1) = 0;
  TEST_C(3, 2) = 3;
  TEST_C(3, 3) = 7;


  Matrix TEST_D(4, 4);

  TEST_D(0, 0) = 4;
  TEST_D(0, 1) = 1;
  TEST_D(0, 2) = 0;
  TEST_D(0, 3) = 0;

  TEST_D(1, 0) = 1;
  TEST_D(1, 1) = 5;
  TEST_D(1, 2) = 4;
  TEST_D(1, 3) = 0;

  TEST_D(2, 0) = 0;
  TEST_D(2, 1) = 4;
  TEST_D(2, 2) = 6;
  TEST_D(2, 3) = -2;

  TEST_D(3, 0) = 0;
  TEST_D(3, 1) = 0;
  TEST_D(3, 2) = -2;
  TEST_D(3, 3) = -8;

  Matrix TEST_E(4, 4);

  TEST_E(0, 0) = 4;
  TEST_E(0, 1) = 2;
  TEST_E(0, 2) = 0;
  TEST_E(0, 3) = 0;

  TEST_E(1, 0) = 2;
  TEST_E(1, 1) = 5;
  TEST_E(1, 2) = 2;
  TEST_E(1, 3) = 0;

  TEST_E(2, 0) = 0;
  TEST_E(2, 1) = 2;
  TEST_E(2, 2) = 5;
  TEST_E(2, 3) = 2;

  TEST_E(3, 0) = 0;
  TEST_E(3, 1) = 0;
  TEST_E(3, 2) = 2;
  TEST_E(3, 3) = 7;

  Matrix TEST_F(4, 4);

  TEST_F(0, 0) = 20;
  TEST_F(0, 1) = -18.43909;
  TEST_F(0, 2) = 0;
  TEST_F(0, 3) = 0;

  TEST_F(1, 0) = -18.43909;
  TEST_F(1, 1) = 41.47059;
  TEST_F(1, 2) = -20.27957;
  TEST_F(1, 3) = 4;

  TEST_F(2, 0) = 4;
  TEST_F(2, 1) = -20.27957;
  TEST_F(2, 2) = 48.38693;
  TEST_F(2, 3) = -24.88061;

  TEST_F(3, 0) = 0;
  TEST_F(3, 1) = 0;
  TEST_F(3, 2) = -24.88061;
  TEST_F(3, 3) = 29.14248;

  Matrix TEST_B_Plus(8, 8);

  TEST_B_Plus(0, 0) = 4;
  TEST_B_Plus(0, 1) = 1;
  TEST_B_Plus(0, 2) = -2;
  TEST_B_Plus(0, 3) = 2;
  TEST_B_Plus(0, 4) = 4;
  TEST_B_Plus(0, 5) = 1;
  TEST_B_Plus(0, 6) = -2;
  TEST_B_Plus(0, 7) = 22;

  TEST_B_Plus(1, 0) = 1;
  TEST_B_Plus(1, 1) = 2;
  TEST_B_Plus(1, 2) = 0;
  TEST_B_Plus(1, 3) = -1;
  TEST_B_Plus(1, 4) = 24;
  TEST_B_Plus(1, 5) = -1;
  TEST_B_Plus(1, 6) = -22;
  TEST_B_Plus(1, 7) = 2;


  TEST_B_Plus(2, 0) = -2;
  TEST_B_Plus(2, 1) = 0;
  TEST_B_Plus(2, 2) = 37;
  TEST_B_Plus(2, 3) = -12;
  TEST_B_Plus(2, 4) = 24;
  TEST_B_Plus(2, 5) = 31;
  TEST_B_Plus(2, 6) = 2;
  TEST_B_Plus(2, 7) = 2;


  TEST_B_Plus(3, 0) = -2;
  TEST_B_Plus(3, 1) = -1;
  TEST_B_Plus(3, 2) = 2;
  TEST_B_Plus(3, 3) = 1;
  TEST_B_Plus(3, 4) = -98;
  TEST_B_Plus(3, 5) = 76;
  TEST_B_Plus(3, 6) = 32;
  TEST_B_Plus(3, 7) = 12;


  TEST_B_Plus(4, 0) = 6;
  TEST_B_Plus(4, 1) = 2;
  TEST_B_Plus(4, 2) = 2;
  TEST_B_Plus(4, 3) = 1;
  TEST_B_Plus(4, 4) = -4;
  TEST_B_Plus(4, 5) = 0;
  TEST_B_Plus(4, 6) = -9;
  TEST_B_Plus(4, 7) = -2;


  TEST_B_Plus(5, 0) = 4;
  TEST_B_Plus(5, 1) = 5;
  TEST_B_Plus(5, 2) = -3;
  TEST_B_Plus(5, 3) = 1;
  TEST_B_Plus(5, 4) = -4;
  TEST_B_Plus(5, 5) = -8;
  TEST_B_Plus(5, 6) = -34;
  TEST_B_Plus(5, 7) = 5;


  TEST_B_Plus(6, 0) = 3;
  TEST_B_Plus(6, 1) = -2;
  TEST_B_Plus(6, 2) = 7;
  TEST_B_Plus(6, 3) = 6;
  TEST_B_Plus(6, 4) = 5;
  TEST_B_Plus(6, 5) = -3;
  TEST_B_Plus(6, 6) = 2;
  TEST_B_Plus(6, 7) = 1;


  TEST_B_Plus(7, 0) = 6;
  TEST_B_Plus(7, 1) = 4;
  TEST_B_Plus(7, 2) = -2;
  TEST_B_Plus(7, 3) = 1;
  TEST_B_Plus(7, 4) = 5;
  TEST_B_Plus(7, 5) = 11;
  TEST_B_Plus(7, 6) = -12;
  TEST_B_Plus(7, 7) = 7;

//TEST DNC - Odd
  Matrix TEST_Odd (3, 3);

  TEST_Odd(0, 0) = 5;
  TEST_Odd(0, 1) = 3;
  TEST_Odd(0, 2) = 0;
  
  TEST_Odd(1, 0) = 3;
  TEST_Odd(1, 1) = 1;
  TEST_Odd(1, 2) = 9;
  
  TEST_Odd(2, 0) = 0;
  TEST_Odd(2, 1) = 9;
  TEST_Odd(2, 2) = 7; 

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

  TEST_S(0, 0) = 6;
  TEST_S(0, 1) = -2;
  TEST_S(1, 0) = -2;
  TEST_S(1, 1) = 3;

  //TEST Gramm-Schmidt
  Matrix TEST_GS (3, 3);

  TEST_GS(0, 0) = 1;
  TEST_GS(0, 1) = -2;
  TEST_GS(0, 2) = 0;
  
  TEST_GS(1, 0) = 2;
  TEST_GS(1, 1) = 2;
  TEST_GS(1, 2) = 0;
  
  TEST_GS(2, 0) = 3;
  TEST_GS(2, 1) = 3;
  TEST_GS(2, 2) = 0; 

  //TEST Gramm-Schmidt - Large
  Matrix TEST_GSL (4, 4);

  TEST_GSL(0, 0) = 7;
  TEST_GSL(0, 1) = 3;
  TEST_GSL(0, 2) = 0;
  TEST_GSL(0, 3) = 0;

  TEST_GSL(1, 0) = 3;
  TEST_GSL(1, 1) = 1;
  TEST_GSL(1, 2) = 0;
  TEST_GSL(1, 3) = 0;

  TEST_GSL(2, 0) = 5;
  TEST_GSL(2, 1) = 2;
  TEST_GSL(2, 2) = 0;
  TEST_GSL(2, 3) = 0;

  TEST_GSL(3, 0) = 3;
  TEST_GSL(3, 1) = 2;
  TEST_GSL(3, 2) = 0;
  TEST_GSL(3, 3) = 0;

  //TEST SVD, S and U_construction - book

  Matrix TEST_U_INIT (5, 3);

  TEST_U_INIT(0, 0) = 1;
  TEST_U_INIT(0, 1) = 0;
  TEST_U_INIT(0, 2) = 1;

  TEST_U_INIT(1, 0) = 0;
  TEST_U_INIT(1, 1) = 1;
  TEST_U_INIT(1, 2) = 0;

  TEST_U_INIT(2, 0) = 0;
  TEST_U_INIT(2, 1) = 1;
  TEST_U_INIT(2, 2) = 1;

  TEST_U_INIT(3, 0) = 0;
  TEST_U_INIT(3, 1) = 1;
  TEST_U_INIT(3, 2) = 0;

  TEST_U_INIT(4, 0) = 1;
  TEST_U_INIT(4, 1) = 1;
  TEST_U_INIT(4, 2) = 0;
  //
  
  /*
    lambda_1 ≈ 0.816
    lambda_2 ≈ 23.994
    lambda_3 ≈ 80.398
    lambda_4 ≈ 177.866
    lambda_5 ≈ 525.760
    lambda_6 ≈ 1779.200
    lambda_7 ≈ 3320.923
    lambda_8 ≈ 17303.044
  */

  
  // 07/04 -tridiagonalization if fully ready
  //trigiagonalization(TEST_B);
  //print_matrix(TEST_B);
  
  //Eigen decomp test for 2 by 2 works
  //Eigen decomp for 4 by 4 works
  //Eigen decomp works for 3 by 3 
  //S_construction works
  //Column_extrac/immerse works
  //Magnitude works
  //populate_matrix works
  //Gramm-Schmidt works - fix paper
  //U_constrction works
  //singular_value_decomp works
  //matrix sorts works - moved here
  //initial_e_approx - works, based on tests, given the diagonal precondition
  //Eigenvector extraction - works

  Matrix A = TEST_E.transpose() * TEST_E;
  print_matrix(A);
  tridiagonalization(A);
  print_matrix(A);
  auto [Eve, Eva] = eigen_decomp(TEST_D);
  print_matrix(Eva);
  //singular_value_decomp(TEST_Odd, Eva, Eve);
  
}


MatrixTuple singular_value_decomp(const Matrix &Init)
{

  Matrix Sym =  Init.transpose() * Init;
  tridiagonalization(Sym);
  auto [Eve, Eva] = eigen_decomp(Sym);

  Matrix S = s_construction(Init, Eva);
  Matrix U = u_construction(Init, Eve, S);

  print_matrix(U);
  print_matrix(S);
  print_matrix(Eve.transpose());

  return std::make_tuple(U, S, Eve.transpose());
}




/*
trigiagonalization - Implementation of Householder's method (Ref. 1 page 602) of turning a
symmetric matrix into a symmetric tridiagonal matrix. Modifies the original by refernce,
uses a helper routine for the similarity transformation matrix production.
*/


void tridiagonalization(Matrix &Sym) {

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
    double v12 = ((l1 - a) / c);
    double v22 = ((l2 - a) / c);
    double v1m = sqrt( 1 + pow( v12, 2));
    double v2m = sqrt( 1 + pow( v22, 2));

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

    //Matrix Hi = Sym.cut( n / 2, 1);
    //Matrix Lo = Sym.cut(n - (n / 2), 0);    
    
    Matrix Hi = Sym.cut( n / 2, 1);
    Matrix Lo = Sym.cut(n / 2, 0); 

    const auto & [Orth1, Diag1] = eigen_decomp(Hi);
    const auto & [Orth2, Diag2] = eigen_decomp(Lo);

    Matrix Orth  = Matrix::combine (Orth1, Orth2);
    Matrix Diag   = Matrix::combine (Diag1, Diag2);

    
    auto OrthT = Orth.transpose();
    const auto & [scalar, unitVector] = Cor;
    Matrix Z = (1 / (sqrt(2))) * (OrthT * unitVector);
      

    /*
    check for the decomposition correctness
    auto corr = Cor.first * ( Cor.second * Cor.second.transpose());
    auto sec = Diag + corr;
    auto thir = Orth * sec * OrthT;

    cout <<"This has to be equal the original\n";
    print_matrix(thir);
    */

    // Fixing order of diagonal and orthogonal entries.
    
    Z = 1 * Z;
    Diag = 1 * Diag;
    double rho = 1 / (2 * scalar);

    if (rho < 0)
    {
      rho = -rho;
      Z = -1 * Z;
      Diag = -1 * Diag;

      auto [D, U] = sorts<true, true>(Diag, Z); 
      Cor = std::make_pair(rho, U);
      Matrix Eval = secular_solver(D, Cor);
      Matrix Evec = evector_extract(Eval, D);

      Eval = -1 * Eval;
      Evec = -1 * Evec;

      auto [Eva, Eve] = sorts<false, false>(Eval, Evec); 
      return MatrixPair(Eve, Eva);
    }
    
    auto [D, U] = sorts<true, true>(Diag, Z); 
    Cor = std::make_pair(rho, U);

    cout << "Rho: "<< rho <<  "\n" ;
    cout << "Diag: \n" ;
    print_matrix(D);
    
    // Retrieving eigenvalues from secular equation.
    Matrix Eval = secular_solver(D, Cor);
    cout << "Eigenvalues:\n";
    print_matrix(Eval);
    // Calculating eigenvectors from defined eigenvalues.
    Matrix Evec = evector_extract(Eval, D);
    cout << "Eigenvector:\n";
    print_matrix(Evec);
    // Fixing order of diagonal and orthogonal entries.
    auto [Eva, Eve] = sorts<false, false>(Eval, Evec); 
      
    return MatrixPair(Eve, Eva);
  }
}




Matrix secular_solver(Matrix &Diag, const Correction &Cor)
{
  const double accuracy = std::pow(10, -8);
  const int m = Diag.rows();
  const auto & [rho, Z] = Cor;
  Matrix Y (m, m);

  
  //defining a matrix with initial eigenvalue approximations:
  Y = initial_e_approx(Diag, Cor);

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


Matrix evector_extract(const Matrix &Eig, const Matrix &Diag)
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
  
  //Matrix of initial approximations
  Matrix Y (m, m);

  for (int k = 0; k < m - 1; ++k)
  {
    const double delta = Diag(k + 1, k + 1) - Diag(k, k);
    const double mid = (Diag(k + 1, k + 1) + Diag(k, k)) / 2;
    const double f = secular_function(Diag, Z, rho, mid);
    double a = 0;
    double b = 0;
    const double c = g_function(Diag, Z, rho, mid, k);
    //buffer counter
    int K = 0;

    if (f >= 0)
    {
      K = k;
      const double z = Z(k, 0);
      const double z_next = Z(k + 1, 0);

      a = (c * delta) + ((z * z) + (z_next * z_next));
      b = z * z * delta;
    }

    else 
    {
      
      K = k + 1;
      const double z = Z(k, 0);
      const double z_next = Z(k + 1, 0);

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
  const double f = secular_function(Diag, Z, rho, mid);
  cout << "f: " << f << "\n";
  if ((f <= 0) && (g <= -h))
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
*/
Matrix u_construction(const Matrix &Init, Matrix &Orth, const Matrix &S)
{
  const int n = Init.rows(); 
  const int m = Init.colms(); // Dimensions of ortho are m x m

  Matrix U (n, n);
  
  #pragma omp parallel for
  for (int i = 0; i < m; ++i)
  {
    double s = S(i, i);
    Matrix C = (1 / s) * Init * Orth.column_extract(i);
    U = Matrix::column_immerse(C, U, i);
  }

  gram_schmidt(U, m);
  
  return U;
}



void gram_schmidt(Matrix &U, int i)
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

  /*
  diagSort - sorts the Matrice's main diagonal in an ascending order.
  Is used in the Secular Equation solver routine
  */
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






double secular_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y)
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

double secular_function_prime(const Matrix& Diag, const Matrix& Z, const double y)
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


double psi(const Matrix& Diag, const Matrix& Z, const double y, const int k)
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

double phi(const Matrix& Diag, const Matrix& Z, const double y, const int k)
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

double psi_prime(const Matrix& Diag, const Matrix& Z, const double y, const int k)
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

double phi_prime(const Matrix& Diag, const Matrix& Z, const double y, const int k)
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

double g_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
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

double h_function(const Matrix& Diag, const Matrix& Z, const double y, const int k)
{
  double z1 = Z(k, 0);
  double z2 = Z(k + 1, 0);
  double sum1 = ((z1 * z1) / (Diag(k, k) - y));
  double sum2 = ((z2 * z2) / (Diag(k + 1, k + 1) - y));
  cout << "here " << Diag(k + 1, k + 1) << "\n";
  cout << "here " << Diag(k, k) << "\n";

  return sum1 + sum2;
}









/*
//finding the upper limit for eigenvalues:
  const double d_out = Diag(j, j) + ((Z.transpose() * Z)(0, 0) / rho);
  const double mid = (d_out + Diag(j, j)) / 2;
  const double g = g_function(Diag, Z, rho, mid, j);
  const double h = h_function(Diag, Z, d_out, j);
  const double f = secular_function(Diag, Z, rho, mid);
  cout << "g: " << g << "\n";
  cout << "h: " << -h << "\n";


  cout << "oh: " << (Z.transpose() * Z)(0, 0) / rho << "\n";

  if ((f <= 0) && (g <= -h))
  {
    Y(j, j) = Diag(j, j) + ((Z.transpose() * Z)(0, 0) / rho);
  }

  else
  {
    const double c = g;
    const double delta = Diag(j, j) - d_out;
    const double z = Z(j + 1, 0);
    const double z_prev = Z(j, 0);
    const double a = (c * delta) + ((z * z) + (z_prev * z_prev));
    const double b = z * z * delta;

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));
    cout << "1: " << d_out +  ((a - root) / (2 * c)) << "\n";
    cout << "2: " << d_out + ((2 * b) / (a + root)) << "\n";

    if (a >= 0)
    {
      //Making an initial approximation y = tau + d_K
      Y(j, j) = d_out + ((a + root) / (2 * c));
    }

    else 
    {
      //Making an initial approximation y = tau + d_K
      Y(j, j) = d_out + ((2 * b) / (a - root));
    }

  }
  */