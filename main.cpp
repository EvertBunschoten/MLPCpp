/*!
* \file main.cpp
* \brief Example script demonstrating the use of the MLPCpp library within C++
code.
* \author E.C.Bunschoten
* \version 2.1.0
*
* MLPCpp Project Website: https://github.com/EvertBunschoten/MLPCpp
*
* Copyright (c) 2023 Evert Bunschoten

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
/*--- Include the look-up MLP class ---*/
#include "include/CLookUp_ANN.hpp"
#include <chrono>

using namespace std;

int main() {
  /* PREPROCESSING START */

  /* Step 1: Generate MLP collection */

  /*--- First specify an array of MLP input file names (preprocessing) ---*/
  string input_filenames[] = {
      "MLP_test.mlp"}; /*!< String array containing MLP input file names. */
  unsigned short nMLPs = sizeof(input_filenames) / sizeof(string);

  /*--- Generate a collection of MLPs with the architectures described in the
   * input file(s) ---*/
  MLPToolbox::CLookUp_ANN ANN_test =
      MLPToolbox::CLookUp_ANN(nMLPs, input_filenames);

  double val_u, val_v, val_y, val_dydu, val_dydv, val_d2ydu2, val_d2ydudv, val_d2ydv2;
  /*--- Generate the input-output map and pair the loaded MLP's with the input
   * and output variables of the lookup operation ---*/
  MLPToolbox::CIOMap iomap = MLPToolbox::CIOMap();

  iomap.AddQueryInput("u", &val_u);
  iomap.AddQueryInput("v", &val_v);
  iomap.AddQueryOutput("y", &val_y);
  iomap.AddQueryJacobian("y", "u", &val_dydu);
  iomap.AddQueryJacobian("y", "v", &val_dydv);
  iomap.AddQueryHessian("y", "u", "u", &val_d2ydu2);
  iomap.AddQueryHessian("y", "u", "v", &val_d2ydudv);
  iomap.AddQueryHessian("y", "v", "v", &val_d2ydv2);
  
  MLPToolbox::CIOMap iomap_output_only = MLPToolbox::CIOMap();
  iomap_output_only.AddQueryInput("u", &val_u);
  iomap_output_only.AddQueryInput("v", &val_v);
  iomap_output_only.AddQueryOutput("y", &val_y);

  ANN_test.PairVariableswithMLPs(iomap);
  ANN_test.PairVariableswithMLPs(iomap_output_only);
  /*--- Optional: display network architecture information in the terminal ---*/
  ANN_test.DisplayNetworkInfo();

  /* PREPROCESSING END */

  /* Step 3: Evaluate MLPs (in iterative process)*/

  ifstream input_data_file;
  ofstream output_data_file;
  string line, word;
  input_data_file.open("reference_data.csv");
  output_data_file.open("predicted_data.csv");
  getline(input_data_file, line);
  output_data_file << line << endl;

  cout << "Derivative finite-differences, Analytical derivative"<<endl;
  
  while (getline(input_data_file, line)) {
    stringstream line_stream(line);
    line_stream >> val_u;
    line_stream >> val_v;

    ANN_test.Predict(iomap);

    output_data_file << scientific << val_u << "\t"
                     << scientific << val_v << "\t"
                     << scientific << val_y << endl;

    /* Validate gradient computation */
    double delta_CV = 1e-5;
    double val_output_p, val_output_m;
    val_u += delta_CV;
    ANN_test.Predict(iomap_output_only);

    val_output_p = val_y;
    val_u -= 2*delta_CV;
    ANN_test.Predict(iomap_output_only);
    val_output_m = val_y;
    double dy_du_fd = (val_output_p - val_output_m) / (2*delta_CV);
    cout << scientific << val_dydu << "\t" << scientific << dy_du_fd << endl;
  }
  input_data_file.close();
  output_data_file.close();
  
}
