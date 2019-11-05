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



/************************************************************/
// Using declarations

using std::cin;
using std::cout;
using std::endl;
using std::pow;
using std::vector;
using std::sqrt;
using std::pow;


/************************************************************/
// Function prototypes/global vars/typedefs

double func(double x);

void regulaFalsi(double a, double b);

/************************************************************/
// Main function

int main(int argc, char *argv[]) {

regulaFalsi(-1, 1);
}




double func(double x) 
{ 
    return (pow(x, 3)) + (4 * x) + 2;
} 

void regulaFalsi(double a, double b) 
{ 
    if (func(a) * func(b) >= 0) 
    { 
        cout << "You have not assumed right a and b\n"; 
        return; 
    } 
  
    double c = a;  // Initialize result 
  
    for (int i=0; i < 10; i++) 
    { 
        // Find the point that touches x axis 
        c = (a*func(b) - b*func(a))/ (func(b) - func(a)); 
  
        // Check if the above found point is root 
        if (func(c)==0) 
            break; 
  
        // Decide the side to repeat the steps 
        else if (func(c)*func(a) < 0) 
            b = c; 
        else
            a = c; 
    } 
    cout << "The value of root is : " << c; 
} 