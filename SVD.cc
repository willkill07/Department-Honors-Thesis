/*
  Author     : Maksim Y. Melnichenko
  Title      : SVD Inplementation

  Project history and additional details are avaliable upon request.
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

struct Node {
  Node* left;
  Node* right;
  std::pair<Matrix, Matrix> data;

  Node (const Matrix& m1, const Matrix& m2)
  : left(nullptr)
  , right(nullptr)
  , data (m1, m2) {}
};


/************************************************************/
// Function prototypes/global vars/typedefs

void build(Matrix &A);

void tridiagonalizer(Matrix &B);

void similarityProducer(Matrix &WTW, double RSQ);

Node* divideNConquer(Matrix &B);

Correction block_diagonal(Matrix &B);

Matrix secular_solver(const Matrix &diag, Correction Beta);

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
  tridiagonalizer(TEST);
}


/*
Build - by reference, modifies the original matrix. Assignes a random
floating point number to each of its cells.
*/
void build(Matrix &A) {
  
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
Tridiagonalizer - Implementation of Householder's method of turning a
symmetric matrix into a symmetric tridiagonal one. Modifies the original by refernce,
uses a helper routine for the similarity transformation matrix production.
*/
void tridiagonalizer(Matrix &B) {
  // producing a column vector, replicating the last column of B

  for (int k = 0; k < B.rows() - 2; ++k) {

    double alpha, RSQ = 0;
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

    B = WTW * B * WTW;
    printMatrix(B);
  }
}



/*
similarityProducer - by reference, takes in the matrix, which will further be used as a 
basis for the similarity transformation matrix.
*/
void similarityProducer(Matrix &WTW, double RSQ) {
  for (int i = 0; i < WTW.rows(); ++i) 
  {
    for (int j = 0; j < WTW.colms(); ++j) 
    {
      WTW(i, j) = -(4 / RSQ) * WTW(i, j);
    }
    WTW(i, i) += 1;
  }
}




/*
Divide - initial step of Cuppen's Divide and Conquer Eigenvalue Extraction algorithm.
/////////////////////////////////UNDER CONSTRUCTION/////////////////////////////////
*/
Node* divideNConquer(Matrix &B)
{

  int n = B.rows();
  Correction Beta = block_diagonal(B);

  if (n == 2) 
  {


    double a  = B(0, 0);
    double d = B(1, 1);
    double c  = B(1, 0);
    double l1, l2;

    Matrix ortho (n, n);
    Matrix diag  (n, n);

    l1, diag(0, 0) = ((a + d) / 2) + sqrt( pow((a + d), 2) - ((a * d) - pow(c, 2)));
    l2, diag(1, 1) = ((a + d) / 2) - sqrt( pow((a + d), 2) - ((a * d) - pow(c, 2)));

    ortho(0, 0), ortho(0, 1) = 1;
    ortho(1, 0) = (l1 - d) / c;
    ortho(1, 1) = (l2 - d) / c;

    Node* data = new Node(ortho, diag);

    return data;
  } 
  else 
  {

    Matrix hi = B.cut( n / 2, 1);
    Matrix lo = B.cut(n - (n / 2), 0);

    Node* hiNode = divideNConquer(hi);
    Node* loNode = divideNConquer(lo);

    Matrix ortho  = Matrix::combine ((hiNode -> data).first, (loNode -> data).first);
    Matrix orthoT = Matrix::combine (((hiNode -> data).first).transpose(), ((loNode -> data).first).transpose());
    Matrix diag   = Matrix::combine ((hiNode -> data).second, (loNode -> data).second);
    Matrix C = (1 / (sqrt(2))) * (orthoT * Beta.second);
    Beta = std::make_pair(2 * Beta.first, C);

    delete hiNode;
    delete loNode;
    return new Node (ortho, secular_solver(diag, Beta));
  }
}




Matrix secular_solver(const Matrix &D, Correction Beta)
{

  double n = D.rows();
  double e = pow(10, -14);
  double sumN, sumD, total;
  Matrix Z = Beta.second;
  double p = Beta.first;

  //setting up initial approximation for eigenvalues
  Matrix l (n, 0);

  for(int i = 0; i < n; ++i)
  {
    if(i == (n - 1))
    {
      double z = (Z * Z.transpose())(0, 0);
      l(i, 0) = (p * z) / 2;
    }
    else
    {
      l(i, 0) = (l(i + 1, 0) - l(i, 0)) / 2;
    }
  }


  for(int i = 0; i < n; ++i)
  {
    do
    {
      sumN, sumD, total = 0;
      for(int j = 0; j < n; ++j)
      {
        sumN = p * (pow(Z(j, 0), 2) / (D(j, j) - l(i, 0)));
        sumD = - (p * (pow(Z(j, 0), 2) / (pow((D(j, j) - l(i, 0)), 2))));
      }

      total = (1 + sumN) / sumD;
      l(i, 0) += total;

    } while (total > e);
  }

  return l;
}



/*
block_diagonal - routine that makes the original matrix, taken in by reference, 
block diagonal and additionally updates the "factored-out" matrix beta with corresponding
elements.
*/

Correction
block_diagonal(Matrix &B)
{
  int n = B.rows();
  Matrix Beta (n, 1);

  double m = n / 2;
  double beta_value = B(m, m - 1);

  Beta(m, 1) = Beta(m - 1, 1) = beta_value;

  B(m, m - 1) = B(m - 1, m) = 0;
  B(m, m) -= beta_value;
  B(m - 1, m - 1) -= beta_value;

  return std::make_pair(beta_value, Beta);
}


/*
Printing - Simple routine, created for the testing purposes.
*/
void printMatrix(const Matrix A) 
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
