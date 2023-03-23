/*!
 * \file ReadNeuralNetwork.cpp
 * \brief Implementation of the reader class to read .mlp input files
 *      used to set up multi-layer perceptrons.
 * \author E.C.Bunschoten
 * \version 1.0.0
 */

#include "../include/CReadNeuralNetwork.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace std;

MLPToolbox::CReadNeuralNetwork::CReadNeuralNetwork(string filename_in) {
  filename = filename_in;
}

void MLPToolbox::CReadNeuralNetwork::ReadMLPFile() {
  ifstream file_stream;
  file_stream.open(filename.c_str(), ifstream::in);
  if (!file_stream.is_open()) {
    throw std::invalid_argument("There is no MLP file called " + filename);
  }

  string line, word;
  bool eoHeader = false, found_layercount = false, found_input_names = false,
       found_output_names = false;

  /* Read general architecture information from file header */

  line = SkipToFlag(&file_stream, "<header>");

  while (getline(file_stream, line) && !eoHeader) {
    /* Read layer count */
    if (line.compare("[number of layers]") == 0) {
      getline(file_stream, line);
      n_layers = stoul(line);
      n_neurons.resize(n_layers);
      biases_mat.resize(n_layers);
      weights_mat.resize(n_layers - 1);
      activation_functions.resize(n_layers);

      found_layercount = true;
    }

    /* Set number of neurons for each layer */
    if (line.compare("[neurons per layer]") == 0) {
      /* In case layer count was not yet provided, return an error */
      if (!found_layercount) {
        throw std::invalid_argument(
            "No layer count provided before defining neuron count per layer");
      }
      /* Loop over layer count and size neuron count and bias count per layer
       * accordingly */
      for (auto iLayer = 0u; iLayer < n_layers; iLayer++) {
        getline(file_stream, line);
        n_neurons[iLayer] = stoul(line);
        biases_mat[iLayer].resize(n_neurons[iLayer]);
      }
      /* Loop over spaces between layers and size the weight matrices
       * accordingly */
      for (auto iLayer = 0u; iLayer < n_layers - 1; iLayer++) {
        weights_mat[iLayer].resize(n_neurons[iLayer]);
        for (auto iNeuron = 0u; iNeuron < n_neurons[iLayer]; iNeuron++)
          weights_mat[iLayer][iNeuron].resize(n_neurons[iLayer + 1]);
      }
      /* Size input and output normalization and set default values */
      input_norm.resize(n_neurons[0]);
      for (auto iNeuron = 0u; iNeuron < n_neurons[0]; iNeuron++)
        input_norm[iNeuron] = make_pair(0, 1);

      output_norm.resize(n_neurons[n_neurons.size() - 1]);
      for (auto iNeuron = 0u; iNeuron < n_neurons[n_neurons.size() - 1];
           iNeuron++)
        output_norm[iNeuron] = make_pair(0, 1);
    }

    /* Read layer activation function types */
    if (line.compare("[activation function]") == 0) {
      if (!found_layercount) {
        throw std::invalid_argument("No layer count provided before providing "
                                    "layer activation functions");
      }
      for (auto iLayer = 0u; iLayer < n_layers; iLayer++) {
        getline(file_stream, line);
        istringstream activation_stream(line);
        activation_stream >> word;
        activation_functions[iLayer] = word;
      }
    }

    /* Read MLP input variable names */
    if (line.compare("[input names]") == 0) {
      found_input_names = true;
      input_names.resize(n_neurons[0]);
      for (auto iInput = 0u; iInput < n_neurons[0]; iInput++) {
        getline(file_stream, line);
        input_names[iInput] = line;
      }
    }

    /* In case input normalization is applied, read upper and lower input bounds
     */
    if (line.compare("[input normalization]") == 0) {
      for (auto iInput = 0u; iInput < input_norm.size(); iInput++) {
        getline(file_stream, line);
        if (line.compare("") != 0) {
          istringstream input_norm_stream(line);
          input_norm_stream >> word;
          double input_min = stold(word);
          input_norm_stream >> word;
          double input_max = stold(word);
          input_norm[iInput] = make_pair(input_min, input_max);
        }
      }
    }

    /* Read MLP output variable names */
    if (line.compare("[output names]") == 0) {
      found_output_names = true;
      auto n_outputs = n_neurons[n_neurons.size() - 1];
      output_names.resize(n_outputs);
      for (auto iOutput = 0u; iOutput < n_outputs; iOutput++) {
        getline(file_stream, line);
        output_names[iOutput] = line;
      }

      if (output_names.size() != (n_neurons[n_neurons.size() - 1])) {
        throw std::invalid_argument("No layer count provided before providing "
                                    "layer activation functions");
      }
    }

    /* In case output normalization is applied, read upper and lower output
     * bounds */
    if (line.compare("[output normalization]") == 0) {
      for (auto iOutput = 0u; iOutput < output_norm.size(); iOutput++) {
        getline(file_stream, line);
        if (line.compare("") != 0) {
          istringstream output_norm_stream(line);
          output_norm_stream >> word;
          double output_min = stold(word);
          output_norm_stream >> word;
          double output_max = stold(word);
          output_norm[iOutput] = make_pair(output_min, output_max);
        }
      }
    }

    if (line.compare("</header>") == 0) {
      eoHeader = true;
    }
  } // eoHeader

  /* Error checking */
  if (!found_input_names) {
    throw std::invalid_argument("No MLP input variable names provided");
  }
  if (!found_output_names) {
    throw std::invalid_argument("No MLP input variable names provided");
  }

  /* Read weights for each layer */
  line = SkipToFlag(&file_stream, "[weights per layer]");
  for (auto iLayer = 0u; iLayer < n_layers - 1; iLayer++) {
    getline(file_stream, line);
    for (auto iNeuron = 0u; iNeuron < n_neurons[iLayer]; iNeuron++) {
      getline(file_stream, line);
      istringstream weight_stream(line);
      for (auto jNeuron = 0u; jNeuron < n_neurons[iLayer + 1]; jNeuron++) {
        weight_stream >> word;
        weights_mat[iLayer][iNeuron][jNeuron] = stold(word);
      }
    }
    getline(file_stream, line);
  }

  /* Read biases for each neuron */
  line = SkipToFlag(&file_stream, "[biases per layer]");
  for (auto iLayer = 0u; iLayer < n_layers; iLayer++) {
    getline(file_stream, line);
    istringstream bias_stream(line);
    for (auto iNeuron = 0u; iNeuron < n_neurons[iLayer]; iNeuron++) {
      bias_stream >> word;
      biases_mat[iLayer][iNeuron] = stold(word);
    }
  }
}

string MLPToolbox::CReadNeuralNetwork::SkipToFlag(ifstream *file_stream,
                                                  string flag) {
  /*--- Search file for a line and set it as the current line in the file stream
   * ---*/
  string line;
  getline(*file_stream, line);

  while (line.compare(flag) != 0 && !(*file_stream).eof()) {
    getline(*file_stream, line);
  }

  if ((*file_stream).eof())
    cout << "line not in file!" << endl;

  return line;
}