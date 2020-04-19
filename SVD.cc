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

MatrixTuple singular_value_decomp(const Matrix &Init, const Matrix &Eig, Matrix &Orth);

/************************************************************/
// Helper function prototypes

void populate_matrix(Matrix &Init); //

Correction block_diagonal(Matrix &Sym); //

Matrix initial_e_approx(const Matrix &Diag, const Correction &Cor);

Matrix s_construction(const Matrix &Init, const Matrix &Eig); //

Matrix u_construction(const Matrix &Init,  Matrix &Orth, const Matrix &S); //

void gram_schmidt(Matrix &U, int i); //

template <bool O, bool T>
void print_matrix(const MatrixT<O,T> &A); //

/************************************************************/
// Small mathematical functions

double secular_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double secular_function_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y);

double psi(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double phi(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double psi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double phi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double g_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

double h_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k);

template <bool Asc, bool Row, bool T1, bool O1, bool T2, bool O2>
MatrixPair sorts(const MatrixT<T1, O1> &Diag, MatrixT<T2, O2> &Orth);

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
  Matrix TEST_U_S (5, 3);

  TEST_U_S(0, 0) = std::sqrt(5);
  TEST_U_S(0, 1) = 0;
  TEST_U_S(0, 2) = 0;

  TEST_U_S(1, 0) = 0;
  TEST_U_S(1, 1) = std::sqrt(2);
  TEST_U_S(1, 2) = 0;

  TEST_U_S(2, 0) = 0;
  TEST_U_S(2, 1) = 0;
  TEST_U_S(2, 2) = 1;

  TEST_U_S(3, 0) = 0;
  TEST_U_S(3, 1) = 0;
  TEST_U_S(3, 2) = 0;

  TEST_U_S(4, 0) = 0;
  TEST_U_S(4, 1) = 0;
  TEST_U_S(4, 2) = 0;

  Matrix TEST_U_V (3, 3);

  TEST_U_V(0, 0) = std::sqrt(6) / 6;
  TEST_U_V(0, 1) = std::sqrt(3) / 3;
  TEST_U_V(0, 2) = -std::sqrt(2) / 2;

  TEST_U_V(1, 0) = std::sqrt(6) / 3;
  TEST_U_V(1, 1) = -std::sqrt(3) / 3;
  TEST_U_V(1, 2) = 0;

  TEST_U_V(2, 0) = std::sqrt(6) / 6;
  TEST_U_V(2, 1) = std::sqrt(3) / 3;
  TEST_U_V(2, 2) = std::sqrt(2) / 2;

  Matrix TEST_U_E (3, 3);

  TEST_U_E(0, 0) = 5;
  TEST_U_E(0, 1) = 0;
  TEST_U_E(0, 2) = 0;

  TEST_U_E(1, 0) = 0;
  TEST_U_E(1, 1) = 2;
  TEST_U_E(1, 2) = 0;

  TEST_U_E(2, 0) = 0;
  TEST_U_E(2, 1) = 0;
  TEST_U_E(2, 2) = 1;


  //TEST Eigenvector - book

  Matrix TEST_Vec (5, 3);

  TEST_Vec(0, 0) = 1;
  TEST_Vec(0, 1) = 0;
  TEST_Vec(0, 2) = 1;

  TEST_Vec(1, 0) = 0;
  TEST_Vec(1, 1) = 1;
  TEST_Vec(1, 2) = 0;

  TEST_Vec(2, 0) = 0;
  TEST_Vec(2, 1) = 1;
  TEST_Vec(2, 2) = 1;

  TEST_Vec(3, 0) = 0;
  TEST_Vec(3, 1) = 1;
  TEST_Vec(3, 2) = 0;

  TEST_Vec(4, 0) = 1;
  TEST_Vec(4, 1) = 1;
  TEST_Vec(4, 2) = 0;

  Matrix TEST_Vec_diag (3, 3);

  TEST_Vec_diag(0, 0) = 5.68554;
  TEST_Vec_diag(0, 1) = 0;
  TEST_Vec_diag(0, 2) = 0;

  TEST_Vec_diag(1, 0) = 0;
  TEST_Vec_diag(1, 1) = 3.41421;
  TEST_Vec_diag(1, 2) = 0;

  TEST_Vec_diag(2, 0) = 0;
  TEST_Vec_diag(2, 1) = 0;
  TEST_Vec_diag(2, 2) = 1.72867;

  Matrix TEST_Vec_Eigen (3, 3);

  TEST_Vec_Eigen(0, 0) = 5;
  TEST_Vec_Eigen(0, 1) = 0;
  TEST_Vec_Eigen(0, 2) = 0;

  TEST_Vec_Eigen(1, 0) = 0;
  TEST_Vec_Eigen(1, 1) = 2;
  TEST_Vec_Eigen(1, 2) = 0;

  TEST_Vec_Eigen(2, 0) = 0;
  TEST_Vec_Eigen(2, 1) = 0;
  TEST_Vec_Eigen(2, 2) = 1;


  
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
  
  
  //Matrix B = TEST_Vec * TEST_Vec.transpose();
  //tridiagonalization(B);
  ///auto [Z, diag] = eigen_decomp(B);
  
  auto [Z, diag] = eigen_decomp(TEST_L);
  

  //evector_extract(TEST_Vec_Eigen, TEST_Vec_diag);



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
    auto OrthT = Orth.transpose();
    Matrix Diag   = Matrix::combine (Diag1, Diag2);

    const auto & [scalar, unitVector] = Cor;

    Matrix Z = (1 / (sqrt(2))) * (OrthT * unitVector);
    Cor = std::make_pair( 1/(2 * scalar), Z);  // REMEMBER ABOUT RHO

    /*
    check for the decomposition correctness
    auto corr = Cor.first * ( Cor.second * Cor.second.transpose());
    auto sec = Diag + corr;
    auto thir = Orth * sec * OrthT;

    cout <<"This has to be equal the original\n";
    print_matrix(thir);
    */



    // Fixing order of diagonal and orthogonal entries.


    auto[D, O] = sorts<true, true>(Diag, Z); 
    print_matrix(D);
    Matrix E = secular_solver(D, Cor);
    print_matrix(E);

    // Retrieving eigenvalues from secular equation.
    //Matrix Evalue = secular_solver(Diag, Cor);
    // Calculating eigenvectors from defined eigenvalues.
    //Matrix Evector = evector_extract(Evalue, Diag);
    // Fixing order of diagonal and orthogonal entries.
    //Matrix::sorts<1>(Evalue, Evector);
    
    return MatrixPair(Orth, Diag);
    //return MatrixPair (Evector, Evalue);
  }
}





Matrix secular_solver(Matrix &Diag, const Correction &Cor)
{
  const int m = Diag.rows();
  const auto & [rho, Z] = Cor;
  Matrix Y (m, m);

  
    //defining a matrix with initial eigenvalue approximations:
    Y = initial_e_approx(Diag, Cor);
    cout << "APPROX:\n";
    print_matrix(Y);

    //finding the upper limit for eigenvalues:
  for (int cnt = 0; cnt < 20; ++cnt)
  {
    const double d_max = Diag(m - 1, m - 1) + ((Z.transpose() * Z)(0, 0) / rho);

    for (int k = 0; k < m - 1; ++k)
    {
      //small necessary computations
      const double y = Y(k, k);
      const double delta = Diag(k, k) - y;
      const double delta_next = Diag(k + 1, k + 1) - y;
      const double w = secular_function(Diag, Z, rho, y);
      const double w_ = secular_function_prime(Diag, Z, rho, y);
      const double psi = psi(Diag, Z, rho, y, k);
      const double phi = phi(Diag, Z, rho, y, k);
      const double psi_ = psi_prime(Diag, Z, rho, y, k);
      const double phi_ = phi_prime(Diag, Z, rho, y, k);


      const double a = (w * (delta + (phi / phi_))) - (delta * phi * (1 + (psi_ / phi_)));
      const double b = delta * w * (phi / phi_);
      const double c = pho + psi - (delta * psi_);

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


    const double a = ((delta + delta_prev) * w) - (buf * w_);
    const double b = buf * w;
    const double c = w - (delta_prev * psi_) - (Z(k, k) * Z(k, k) / delta);

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));

    if (a >= 0)
    {
      //updating the approximation matrix
      Y(k, k) += (a + root) / (2 * c);
    }

    else
    {
      //updating the approximation matrix
      Y(k, k) += (2 * b) / (a - root);
    }

    cout <<"Update::\n";
    print_matrix(Y);
  }

  return Y;  
}


/*
THE MIDDLE WAY
Matrix secular_solver(Matrix &Diag, const Correction &Cor)
{
  const int m = Diag.rows();
  const auto & [rho, Z] = Cor;
  Matrix Y (m, m);

  
    //defining a matrix with initial eigenvalue approximations:
    Y = initial_e_approx(Diag, Cor);
    cout << "APPROX:\n";
    print_matrix(Y);

    //finding the upper limit for eigenvalues:
  for (int cnt = 0; cnt < 20; ++cnt)
  {
    const double d_max = Diag(m - 1, m - 1) + ((Z.transpose() * Z)(0, 0) / rho);

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

      const double a = ((delta + delta_next) * w) - (buf * w_);
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

    const double a = ((delta + delta_prev) * w) - (buf * w_);
    const double b = buf * w;
    const double c = w - (delta_prev * psi_) - (Z(k, k) * Z(k, k) / delta);

    //saving a computation
    const double root = std::sqrt((a * a) - (4 * b * c));

    if (a >= 0)
    {
      //updating the approximation matrix
      Y(k, k) += (a + root) / (2 * c);
      cout << "FIRST\n";
      cout << (a + root) / (2 * c) <<"\n";
    }

    else
    {
      //updating the approximation matrix
      Y(k, k) += (2 * b) / (a - root);
      cout << "SECOND\n";
    }
    
    cout <<"Update::\n";
    print_matrix(Y);
  }

  return Y;  
}


Matrix evector_extract(const Matrix &Eig, const Matrix &Diag)
{
  const int m = Eig.rows();
  Matrix Evec (m, m);
  Matrix Z (m, 1);

  print_matrix(Z);

  //computing approximation to each z
  //#pragma omp parallel
  for(int i = 0; i < m; ++i)
  {
    auto product1 = 1;
    //#pragma omp for reduction(* : product1)
    for (int j = 0; j < i; ++j)
    {
      product1 *= (Eig(j, j) - Diag(i, i)) / (Diag(j, j) - Diag(i, i));
      cout << product1 <<"\n";
    }

    auto product2 = 1;
    //#pragma omp for reduction(* : product2)
    for (int k = i; k < m; ++k)
    {
      product2 *= (Eig(k, k) - Diag(i, i)) / (Diag(k + 1, k + 1) - Diag(i, i));
      cout << product2 <<"\n";
    }

    auto product3 = Eig(m, m) - Diag(i, i);

    Z(i, 1) = std::sqrt(product1 *product2 * product3);
  }

  print_matrix(Z);

  //computing approximation to each eigenvector
  //#pragma omp parallel
  for(int i = 0; i < m; ++i)
  {
    //#pragma omp for
    for(int j = 0; j < m; ++j)
    {
      Z(j, 1) = Z(j, 1) / (Diag(j, j) - Eig(i, i)); 
    }


    auto sum = 0;
    //#pragma omp for reduction(+ : sum)
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



MatrixTuple singular_value_decomp(const Matrix &Init, const Matrix &Eig, Matrix &Orth)
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
  const double d_max = Diag(m - 1, m - 1) + ((Z.transpose() * Z)(0, 0) / rho);
  
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

      a = c * delta + ((z * z) + (z_next * z_next));
      b = z * z * delta;
    }

    else 
    {
      
      K = k + 1;
      const double z = Z(k, 0);
      const double z_next = Z(k + 1, 0);

      a = -c * delta + ((z * z) + (z_next * z_next));
      b = -z * z * delta;
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
  const double mid = (d_max + Diag(k, k)) / 2;
  const double g = g_function(Diag, Z, rho, mid, k);
  const double h = h_function(Diag, Z, rho, d_max, k);
  const double f = secular_function(Diag, Z, rho, mid);

  if ((g <= -h) && (f <= 0))
  {
    Y(k, k) = Diag(k, k) + ((Z.transpose() * Z)(0, 0) / rho);
  }

  else
  {
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
      //Making an initial approximation y = tau + d_K
      Y(k, k) = Diag(k, k) + ((a + root) / (2 * c));
    }

    else 
    {
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
  double sum;
  double z;

  #pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < m; ++j)
  {
    z = Z(j, 0);
    sum += (z * z) / (Diag(j, j) - y);
  }

  return rho + sum;
}

double secular_function_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y)
{
  const int m = Diag.rows();
  double sum;
  double z;
  double denom; //denominator

  #pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < m; ++j)
  {
    z = Z(j, 0);
    denom = Diag(j, j) - y;
    sum += (z * z) / (denom * denom);
  }

  return sum;
}


double psi(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
{
  double sum;
  double z;
  double denom; //denominator

  #pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < k; ++j)
  {
    z = Z(j, 0);
    sum += (z * z) / ( Diag(j, j) - y;);
  }

  return sum;
}

double phi(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
const int m = Diag.rows();
  double sum;
  double z;
  double denom; //denominator

  #pragma omp parallel for reduction(+ : sum)
  for (int j = k; j < m; ++j)
  {
    z = Z(j, 0);
    sum += (z * z) / (Diag(j, j) - y);
  }

  return sum;
}

double psi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
{
  double sum;
  double z;
  double denom; //denominator

  #pragma omp parallel for reduction(+ : sum)
  for (int j = 0; j < k; ++j)
  {
    z = Z(j, 0);
    denom = Diag(j, j) - y;
    sum += (z * z) / (denom * denom);
  }

  return sum;
}

double phi_prime(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
{
  const int m = Diag.rows();
  double sum;
  double z;
  double denom; //denominator

  #pragma omp parallel for reduction(+ : sum)
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
  double z;

  #pragma omp parallel for reduction(+ : sum)
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

double h_function(const Matrix& Diag, const Matrix& Z, const double rho, const double y, const int k)
{
  double z1 = Z(k, 0);
  double z2 = Z(k + 1, 0);
  double sum1 = ((z1 * z1) / (Diag(k, k) - y));
  double sum2 = ((z2 * z2) / (Diag(k + 1, k + 1) - y));

  return sum1 + sum2;
}
