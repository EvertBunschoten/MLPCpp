/*!
* \file CIOMap.hpp
* \brief Input-output map class definition for the definition of MLP look-up
operations.
* \author E.C.Bunschoten
* \version 1.2.0
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

#include "variable_def.hpp"
#include <string>
#include <vector>
#include <set>
#include "CNeuralNetwork.hpp"
namespace MLPToolbox {

  class InsufficientOutputs: public std::exception 
  {
  public:
      bool issue_with_inputs{false};
      InsufficientOutputs(const bool is_input) {issue_with_inputs = is_input;};
      ~InsufficientOutputs() = default;
      virtual const char *what() const noexcept {
          if (issue_with_inputs){
            return "Not all queries are present in the network inputs";
          } else
            return "Not all queries are present in the network outputs";
      }
  };

  class DuplicateInputs: public std::exception 
  {
    std::vector<std::string> input_names;
  public:
      DuplicateInputs(const std::vector<std::string> &problematic_inputs)
      {
        input_names.resize(problematic_inputs.size());
        std::copy(problematic_inputs.begin(), problematic_inputs.end(), input_names.begin());
      };
      virtual const char *what() const noexcept {

        std::string message = "Network has duplicate inputs: ";
        for (const auto & var : input_names) message += (var + ", ");

        char *cstr = new char[message.size() + 1];
        std::strcpy(cstr, message.c_str());
        return cstr;
      }
  };

  
struct IOMap_Network {
  /*! \brief struct with query information. */

  CNeuralNetwork* MLP; /*! \brief Pointer to network selected for query. */
  std::vector<std::pair<mlpdouble*, mlpdouble*>> input_map; /*! \brief Link between query input and network input nodes. */
  std::vector<std::pair<mlpdouble*,mlpdouble*>> output_map; /*! \brief Link between network output nodes and query output. */
  std::vector<std::pair<const mlpdouble*, const mlpdouble*>> Jacobian_map;  /*! \brief Link between network Jacobians and query Jacobians. */
  std::vector<std::pair<const mlpdouble*, const mlpdouble*>> Hessian_map;   /*! \brief Link between network Hessians and query Hessians. */
  bool evaluate_Jacobian {false}; /*! \brief Evaluate Jacobians while evaluating the network output. */
  bool evaluate_Hessian {false};  /*! \brief Evaluate Hessians while evaluating the network output. */
};

class CIOMap {
    /*! \brief Class used to pair look-up variables with network outputs. */
    private: 
    std::vector<std::pair<std::string, mlpdouble*>> query_input;  
    std::vector<std::pair<std::string, mlpdouble*>> query_output;
    
    std::vector<IOMap_Network> query_network_maps; /*! \brief Query information per network. */

    std::vector<std::pair<std::pair<std::string, std::string>, mlpdouble*>> query_Jacobian; /*! \brief Jacobians to be evaluated. */
    std::vector<std::pair<std::pair<std::string, std::pair<std::string, std::string>>, mlpdouble*>> query_Hessian; /*! \brief Hessians to be evaluated. */

    std::vector<mlpdouble> query_output_vals; /*! \brief Network outputs corresponding to query. */
    std::vector<mlpdouble*> null_outputs;     /*! \brief Pointers to outputs that should return zero. */

    /*!
    * \brief Check whether look-up variable should return zero.
    * \param[in] variable_name_in - Query variable name to check.
    * \returns - if variable is a variant of "NULL", "NONE", or "ZERO"
    */
    bool CheckNull(const std::string & variable_name_in) const {
      std::string variable_name = variable_name_in;
      std::transform(variable_name.begin(), variable_name.end(), variable_name.begin(), [](unsigned char c){ return std::tolower(c);});
      
      if (!variable_name.compare("null") || !variable_name.compare("none") || !variable_name.compare("zero")) {
        return true;
      } else 
        return false;
    }

    /*!
    * \brief Set the value of null variables to zero.
    */
    void SetNullOutputs() const { for (auto nulls : null_outputs) *nulls = mlpdouble(0.0); }

    /*!
    * \brief Evaluate the output, Jacobian, and Hessian of network selected for query.
    * \param[in] mapped_network - query struct
    */
    void NetworkInference(const IOMap_Network &mapped_network) const { 
      if (mapped_network.MLP->CheckInputInclusion()){
        mapped_network.MLP->CalcJacobian(mapped_network.evaluate_Jacobian);
        mapped_network.MLP->CalcHessian(mapped_network.evaluate_Hessian);
        mapped_network.MLP->Predict();
        RetrieveNetworkOutput(mapped_network);
        mapped_network.MLP->CalcJacobian(false);
        mapped_network.MLP->CalcHessian(false);
      }

    }

    /*!
    * \brief Check whether network inputs are unique.
    * \param[in] network_to_check - pointer to network object.
    */
    bool CheckUniqueInputs(const CNeuralNetwork  *network_to_check) const {
      auto network_inputs = network_to_check->GetInputVars();
      std::sort(network_inputs.begin(),network_inputs.end());
      return std::unique(network_inputs.begin(),network_inputs.end()) == network_inputs.end();
    }

    /*!
    * \brief Check whether the network inputs and output are in the query
    * \param[in] network_to_check - pointer to network object.
    * \returns - if query inputs correspond to network inputs and if at least one query output is in the network output variables.
    */
    bool CheckNetworkVariables(const CNeuralNetwork  *network_to_check) {
        const std::vector<std::string> network_inputs = network_to_check->GetInputVars();
        const std::vector<std::string> network_outputs = network_to_check->GetOutputVars();
        bool network_compatible{true};
        for (auto q_in : query_input) {
            auto a = find(network_inputs.begin(), network_inputs.end(), q_in.first);
            if (a == end(network_inputs)){
                network_compatible = false;
            }
        }
        if (network_compatible) {
            bool found_output{false};
            for (std::string var_out : network_outputs) {
                auto loc = std::find_if(query_output.begin(), query_output.end(), [var_out](std::pair<std::string, mlpdouble*>q) {return q.first==var_out;});
                if (loc != query_output.end()) {
                    found_output = true;
                }
            }
            network_compatible = found_output;
        }
        return network_compatible;
    }

    /*!
    * \brief Copy the query input to the network input.
    * \param[in] mapped_network - network object.
    */
    void SetNetworkInputs(const IOMap_Network &mapped_network) const {
      for (const auto &q_in : mapped_network.input_map) *q_in.second = *q_in.first;
    } 

    /*!
    * \brief Copy the query input to the network input.
    * \param[in] mapped_network - pointer to network object.
    * \param[in] vals_input - query input.
    */
    void SetNetworkInputs(const IOMap_Network &mapped_network, const std::vector<mlpdouble> &vals_input) const {
      for (auto iInput=0u; iInput < mapped_network.input_map.size(); iInput++) 
        *mapped_network.input_map[iInput].second = vals_input[iInput];
    }

    /*!
    * \brief Retrieve query output from network output.
    * \param[in] mapped_network - network object.
    */
    void RetrieveNetworkOutput(const IOMap_Network &mapped_network) const {
      for (const auto &q_out : mapped_network.output_map) *q_out.first = *q_out.second;
    }

    /*!
    * \brief Link the query input terms to the mapped network inputs
    */
    void MapInputs(){
        for (auto &mapped_network : query_network_maps) {
          const auto network_input_vars = mapped_network.MLP->GetInputVars();
          for (const auto &q : query_input) {

            auto loc=std::find(network_input_vars.begin(), network_input_vars.end(), q.first);
            if (loc != network_input_vars.end()) {
              const auto ref_query_input = q.second;
              const auto iInput_network = distance(network_input_vars.begin(), loc);
              auto ref_network_input = mapped_network.MLP->InputLayer(iInput_network);
              mapped_network.input_map.push_back(std::make_pair(ref_query_input, ref_network_input));
            }
          }
        }
    }

    /*!
    * \brief Link the query ouput terms to the mapped network outputs
    */
    void MapOutputs() {
      for (const auto &q : query_output) {
        if (CheckNull(q.first)) {
          null_outputs.push_back(q.second);
        } else {
        for (auto &mapped_network : query_network_maps) {
          const auto network_output_vars = mapped_network.MLP->GetOutputVars();
          auto loc = std::find(network_output_vars.begin(), network_output_vars.end(), q.first);
          if (loc != network_output_vars.end()) {
            const auto iOutput = distance(network_output_vars.begin(), loc);
            const auto ref_query_output = q.second;
            const auto ref_network_output = mapped_network.MLP->OutputLayer(iOutput);
            mapped_network.output_map.push_back(std::make_pair(ref_query_output, ref_network_output));          
          }
          }
        }
      }
    }

    /*!
    * \brief Map the query Jacobian terms to the Jaobians of the mapped networks
    */
    void MapJacobians() {
      for (auto &mapped_network : query_network_maps) {
        const auto network_output_vars = mapped_network.MLP->GetOutputVars();
        const auto network_input_vars = mapped_network.MLP->GetInputVars();
        for (const auto &q : query_Jacobian) {
          const auto ref_output_Jacobian = q.second;
          const auto varname_enumerator = q.first.first;
          const auto varname_demominator = q.first.second;
          auto loc_enumerator = std::find(network_output_vars.begin(), network_output_vars.end(), varname_enumerator);
          if (loc_enumerator != network_output_vars.end()) {
            const auto iOutput = distance(network_output_vars.begin(), loc_enumerator);
            auto loc_demoninator = std::find(network_input_vars.begin(), network_input_vars.end(), varname_demominator);
            if (loc_demoninator != network_input_vars.end()) {
              const auto iInput = distance(network_input_vars.begin(), loc_demoninator);
              mapped_network.output_map.push_back(std::make_pair(ref_output_Jacobian, mapped_network.MLP->Jacobian(iOutput, iInput)));
              mapped_network.evaluate_Jacobian=true;
            }
          }
        }
      }
    }

    /*!
    * \brief Map the query Hessian terms to the Hessians of the mapped networks
    */
    void MapHessians() {
      for (auto &mapped_network : query_network_maps) {
        const auto network_output_vars = mapped_network.MLP->GetOutputVars(), 
                   network_input_vars = mapped_network.MLP->GetInputVars();
        for (const auto &q : query_Hessian) {
          const std::string varname_enumerator = q.first.first,
                            varname_demominator_1 = q.first.second.first, 
                            varname_demominator_2 = q.first.second.second;
          const auto ref_output_Hessian = q.second;

          auto loc_enumerator = std::find(network_output_vars.begin(), network_output_vars.end(), varname_enumerator);
          if (loc_enumerator != network_output_vars.end()) {
            const auto iOutput = std::distance(network_output_vars.begin(), loc_enumerator);
            auto loc_demoninator_1 = std::find(network_input_vars.begin(), network_input_vars.end(), varname_demominator_1);
            if (loc_demoninator_1 != network_input_vars.end()) {
              const auto iInput = std::distance(network_input_vars.begin(), loc_demoninator_1);
              auto loc_denominator_2 = std::find(network_input_vars.begin(), network_input_vars.end(), varname_demominator_2);
              if (loc_denominator_2 != network_input_vars.end()) {
                const auto jInput = std::distance(network_input_vars.begin(), loc_denominator_2);
                const auto ref_network_Hessian = mapped_network.MLP->Hessian(iOutput, iInput, jInput);
                mapped_network.output_map.push_back(std::make_pair(ref_output_Hessian, ref_network_Hessian));
                mapped_network.evaluate_Hessian=true;
                mapped_network.evaluate_Jacobian=true;
              }
            }
          }
        }
      }
    }

    public:
    CIOMap()=default;

    CIOMap(const std::vector<std::string> &varnames_in, const std::vector<std::string> &varnames_out) {
      SetQueryInput(varnames_in);
      SetQueryOutput(varnames_out);
    }
    
    const std::vector<IOMap_Network> GetNetworksInQuery() const {return query_network_maps;}

    /*!
    * \brief Define query input variables
    * \param[in] varnames - vector with input names.
    */
    void SetQueryInput(const std::vector<std::string> &varnames) {
      query_input.clear();
      for (auto var : varnames)
        query_input.push_back(std::make_pair(var, nullptr));
    }

    /*!
    * \brief Define query output variables
    * \param[in] varnames - vectory with output names.
    */
    void SetQueryOutput(const std::vector<std::string> &varnames) {
      query_output.clear();
      query_output_vals.resize(varnames.size());
      for (auto iOutput=0u; iOutput<varnames.size(); iOutput++)
        query_output.push_back(std::make_pair(varnames[iOutput], &query_output_vals[iOutput]));
    }
    
    /*!
    * \brief Get outputs for query.
    */
    std::vector<mlpdouble> GetQueryOutput() const {return query_output_vals;}

    /*!
    * \brief Add input variable to query.
    * \param[in] varname - query input variable name.
    * \param[in] ref_input - pointer to query input variable.
    */
    void AddQueryInput(const std::string varname, mlpdouble* ref_input=nullptr) {
      query_input.push_back(std::make_pair(varname, ref_input));
    }
    
    /*!
    * \brief Add output variable to query.
    * \param[in] varname - query output variable name.
    * \param[in] ref_output - pointer to query output variable.
    */
    void AddQueryOutput(const std::string varname, mlpdouble* ref_output=nullptr) {
      query_output.push_back(std::make_pair(varname, ref_output));
    }

    /*!
    * \brief Add Jacobian to query.
    * \param[in] varname_output - output variable for which to calculate Jacobian.
    * \param[in] varname_input - input variable for which to calculate Jacobian.
    * \param[in] ref_output - pointer to Jacobian output.
    */
    void AddQueryJacobian(const std::string varname_output, const std::string varname_input, mlpdouble*ref_output) {
      query_Jacobian.push_back(std::make_pair(std::make_pair(varname_output, varname_input),ref_output));
    }

    /*!
    * \brief Add Hessian to query.
    * \param[in] varname_output - output variable for which to calculate Hessian.
    * \param[in] varname_input_1 - first input variable for which to calculate Hessian.
    * \param[in] varname_input_2 - second input variable for which to calculate Hessian.
    * \param[in] ref_output - pointer to Hessian output.
    */
    void AddQueryHessian(const std::string varname_output, const std::string varname_input_1, const std::string varname_input_2, mlpdouble*ref_output) {
      query_Hessian.push_back(std::make_pair(std::make_pair(varname_output, std::make_pair(varname_input_1, varname_input_2)),ref_output));
    }


    /*!
    * \brief Identify networks with compatible input and output variables for query.
    * \param[in] networks_to_check - vector with pointers to network pointers. 
    */
    void FindNetworksForQuery(const std::vector<CNeuralNetwork*> &networks_to_check) {
        for (auto network_to_check : networks_to_check) {
            if (!CheckUniqueInputs(network_to_check)) {
              throw DuplicateInputs(network_to_check->GetInputVars());
              return;
            }
            if (CheckNetworkVariables(network_to_check)){
                IOMap_Network mapped_network;
                mapped_network.MLP = network_to_check;
                query_network_maps.push_back(mapped_network);
            }
        }
        if (query_network_maps.empty()) {
          throw InsufficientOutputs(true);
        }
        if (!CheckUseOfOutputs()) {
            throw InsufficientOutputs(false);
        }

        MapInputs();
        MapOutputs();
        MapJacobians();
        MapHessians();
    }

    /*!
    * \brief Set the input and evaluate the output of the mapped networks.
    */
    void operator()() const 
    {
      for (const auto &mapped_network : query_network_maps) {
        SetNetworkInputs(mapped_network);
        NetworkInference(mapped_network);
      }
      SetNullOutputs();

    };

    /*!
    * \brief Set the input and evaluate the output of the mapped networks.
    * \param[in] vals_input - query input values.
    * \param[in] refs_output - pointers to query output.
    */
    void operator()(const std::vector<mlpdouble> &vals_input,const std::vector<mlpdouble*> &refs_output) const 
    {
      if (refs_output.size() != query_output_vals.size()){
        throw std::exception();
      }
      for (const auto &mapped_network : query_network_maps) {
        SetNetworkInputs(mapped_network, vals_input);
        NetworkInference(mapped_network);
      }
      SetNullOutputs();

      for (auto iOutput=0u; iOutput<query_output_vals.size(); iOutput++)
        *refs_output[iOutput] = query_output_vals[iOutput];
    };

    /*!
    * \brief Check whether all query outputs are present in the loaded network outputs.
    * \returns whether all query outputs are included
    */
    bool CheckUseOfOutputs() const {
        bool found_all_query_vars {true};
        for (auto &q : query_output) {
          bool found_var{false};
          if (CheckNull(q.first)) {
            found_var = true;
          }
          for (const auto &mapped_network : query_network_maps) {
            const auto vars_network_out = mapped_network.MLP->GetOutputVars();
            auto loc = std::find(vars_network_out.begin(), vars_network_out.end(), q.first);
            if (loc != vars_network_out.end())
              found_var = true;
          }
          if (!found_var){
              std::cout << "Warning! " << q.first << " is not present in the outputs of any of the loaded networks" << std::endl;
              found_all_query_vars = false;
          }
        }
        return found_all_query_vars;
    }

    /*!
    * \brief Display information about the mapped networks in the terminal.
    */
    void DisplayQueryNetworks() const {
      for (const auto &mapped_network : query_network_maps) {
          mapped_network.MLP->DisplayNetwork();
          std::cout << std::endl;
      }
    }

    std::pair<mlpdouble, mlpdouble> GetInputNorm(const std::string varname) {
      mlpdouble val_limit_1{0},val_limit_2{0};
      for (const auto &mapped_network : query_network_maps) {
        auto network_input_names = mapped_network.MLP->GetInputVars();
        auto loc = std::find(network_input_names.begin(), network_input_names.end(), varname);
        size_t iInput = std::distance(network_input_names.begin(), loc);
        auto input_norm = mapped_network.MLP->GetInputNorm(iInput);
        val_limit_1 += input_norm.first;
        val_limit_2 += input_norm.second;
      }
      val_limit_1 /= query_network_maps.size();
      val_limit_2 /= query_network_maps.size();
      return std::make_pair(val_limit_1, val_limit_2);
    }
};
} // namespace MLPToolbox


