/*!
* \file CIOMap.hpp
* \brief Input-output map class definition for the definition of MLP look-up
operations.
* \author E.C.Bunschoten
* \version 2.1.1
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
    * \brief Check whether query input or output variables are unique sets.
    * \param[in] is_input - check query input or output variables.
    */
    void CheckUniqueQueryVars(const bool is_input=true) const {
      std::vector<std::string> query_duplicates, query_vars_copy;
      const auto query_vars = is_input ? query_input : query_output;
      for (auto q : query_vars) {
        if(!CheckNull(q.first)) query_vars_copy.push_back(q.first);
      }
      std::sort(query_vars_copy.begin(), query_vars_copy.end());
      const auto duplicate = std::adjacent_find(query_vars_copy.begin(), query_vars_copy.end());
      if (duplicate != query_vars_copy.end()) {
        std::string msg = "Query contains duplicate ";
        msg += is_input ? "inputs:":"outputs:";
        msg += *duplicate;
        ErrorMessage(msg, "CIOMap::CheckQueryVars");
      }
    }

    /*!
    * \brief Check if there are shared variables between query input and output variables.
    */
    void CheckSharedVars() const {
      std::vector<std::string> query_input_vars, query_output_vars;
      for (auto q_in : query_input) query_input_vars.push_back(q_in.first);
      for (auto q_out : query_output) query_output_vars.push_back(q_out.first);

      std::vector<std::string> shared_vars;
      shared_vars.clear();
      for (auto q : query_output_vars) {
        auto f = std::find(query_input_vars.begin(), query_input_vars.end(), q);
        if (f != query_input_vars.end())
          shared_vars.push_back(*f);
      }
      if (!shared_vars.empty()) {
        std::string msg = "Query has shared variables between input and output: ";
        for (auto q : shared_vars) msg += (q + " ");
        ErrorMessage(msg, "CIOMap::CheckSharedVars");
      }
      
    }

    /*!
    * \brief Check if Jacobian variables are included in the query variables.
    */
    void CheckJacobianQuery() const {

      /* Gather unique Jacobian enumerator and denominator variables from the query. */
      std::vector<std::string> query_Jacobian_y_vars={}, query_Jacobian_x_vars={};
      for (const auto &q_in : query_Jacobian){
        query_Jacobian_y_vars.push_back(q_in.first.first);
        query_Jacobian_x_vars.push_back(q_in.first.second);
      }
      auto unique_Jac_y_vars = std::set(query_Jacobian_y_vars.begin(), query_Jacobian_y_vars.end());
      auto unique_Jac_x_vars = std::set(query_Jacobian_x_vars.begin(), query_Jacobian_x_vars.end());
      
      /* Check whether the unique Jacobian enumerator set contains the query output variable set. */
      for (auto q_out : query_output) {
        auto f_y = std::find(unique_Jac_y_vars.begin(), unique_Jac_y_vars.end(), q_out.first);
        if (f_y != unique_Jac_y_vars.end()){
          unique_Jac_y_vars.erase(f_y);
        }
      }
      /* Check whether the unique Jacobian denominator set contains the query input variable set. */
      for (auto q_in : query_input) {
        auto f_x = std::find(unique_Jac_x_vars.begin(), unique_Jac_x_vars.end(), q_in.first);
        if (f_x != unique_Jac_x_vars.end()) 
          unique_Jac_x_vars.erase(f_x);
      }

      if (!unique_Jac_y_vars.empty() || !unique_Jac_x_vars.empty()) {
        std::string msg = "Jacobian enumerator variables not included in query: ";
        for (auto s : unique_Jac_y_vars) msg += (s + " ");
        msg += "\n";
        msg += "Jacobian denominator  variables not included in query: ";
        for (auto s : unique_Jac_x_vars) msg += (s + " ");
        ErrorMessage(msg, "CIOMap::CheckJacobianQuery");
      }
    }

    /*!
    * \brief Check if Jacobian variables are included in the query variables.
    */
    void CheckHessianQuery() const {

      /* Gather unique Hessian enumerator and denominator variables from the query. */
      std::vector<std::string> query_Hessian_y_vars={}, query_Hessian_x_vars={};
      for (const auto &q_in : query_Hessian){
        query_Hessian_y_vars.push_back(q_in.first.first);
        query_Hessian_x_vars.push_back(q_in.first.second.first);
        query_Hessian_x_vars.push_back(q_in.first.second.second);
      }
      auto unique_Hes_y_vars = std::set(query_Hessian_y_vars.begin(), query_Hessian_y_vars.end());
      auto unique_Hes_x_vars = std::set(query_Hessian_x_vars.begin(), query_Hessian_x_vars.end());

      /* Check whether the unique Hessian enumerator set contains the query output variable set. */
      for (auto q_out : query_output) {
        auto f_y = std::find(unique_Hes_y_vars.begin(), unique_Hes_y_vars.end(), q_out.first);
        if (f_y != unique_Hes_y_vars.end()){
          unique_Hes_y_vars.erase(f_y);
        }
      }
      /* Check whether the unique Hessian denominator set contains the query input variable set. */
      for (auto q_in : query_input) {
        auto f_x = std::find(unique_Hes_x_vars.begin(), unique_Hes_x_vars.end(), q_in.first);
        if (f_x != unique_Hes_x_vars.end()) 
          unique_Hes_x_vars.erase(f_x);
      }

      if (!unique_Hes_y_vars.empty() || !unique_Hes_x_vars.empty()) {
        std::string msg = "Hessian enumerator variables not included in query: ";
        for (auto s : unique_Hes_y_vars) msg += (s + " ");
        msg += "\n";
        msg += "Hessian denominator variables not included in query: ";
        for (auto s : unique_Hes_x_vars) msg += (s + " ");
        ErrorMessage(msg, "CIOMap::CheckHessianQuery");
      }
    }

    /*!
    *\brief Check if the enumerator and denominator of the query Jacobians are in the inputs and outputs of the same network.
    */
    void CheckJacobianNetworks() const { 

      std::vector<std::pair<std::string, std::string>> incompatible_jacobians={};
      for (auto J_q : query_Jacobian) {
        bool compatible_jac{false};
        std::string name_enumerator = J_q.first.first;
        std::string name_denominator = J_q.first.second;
        
        for (auto M : query_network_maps) {
          auto input_vars = M.MLP->GetInputVars();
          auto output_vars = M.MLP->GetOutputVars();
          
          auto f_output = std::find(output_vars.begin(), output_vars.end(), name_enumerator);
          auto f_input = std::find(input_vars.begin(), input_vars.end(), name_denominator);
          if (f_output != output_vars.end() && f_input != input_vars.end())
            compatible_jac = true;
        }

        if (!compatible_jac)
          incompatible_jacobians.push_back(std::make_pair(name_enumerator, name_denominator));
      }

      if (!incompatible_jacobians.empty()) {
        std::string msg = "The following Jacobian queries were not supported by the networks:\n";
        for (auto J : incompatible_jacobians)
          msg += ("d" + J.first + "/d"+J.second+"\n");
        ErrorMessage(msg, "CIOMap::CheckJacobianNetworks");
      }
    }

    /*!
    *\brief Check if the enumerator and denominators of the query Hessians are in the inputs and outputs of the same network.
    */
    void CheckHessianNetworks() const { 

      std::vector<std::pair<std::string, std::pair<std::string, std::string>>> incompatible_hessians={};
      for (auto H_q : query_Hessian) {
        bool compatible_hes{false};
        std::string name_enumerator = H_q.first.first;
        std::string name_denominator_1 = H_q.first.second.first;
        std::string name_denominator_2 = H_q.first.second.second;

        for (auto M : query_network_maps) {
          auto input_vars = M.MLP->GetInputVars();
          auto output_vars = M.MLP->GetOutputVars();
          
          auto f_output = std::find(output_vars.begin(), output_vars.end(), name_enumerator);
          auto f_input_1 = std::find(input_vars.begin(), input_vars.end(), name_denominator_1);
          auto f_input_2 = std::find(input_vars.begin(), input_vars.end(), name_denominator_2);
          
          if (f_output != output_vars.end() && f_input_1 != input_vars.end() && f_input_2 != input_vars.end())
            compatible_hes = true;
        }

        if (!compatible_hes)
          incompatible_hessians.push_back(std::make_pair(name_enumerator, std::make_pair(name_denominator_1,name_denominator_2)));
      }

      if (!incompatible_hessians.empty()) {
        std::string msg = "The following Hessian queries were not supported by the networks:\n";
        for (auto H : incompatible_hessians)
          msg += ("d2" + H.first + "/d"+H.second.first + "d"+H.second.second+"\n");
        ErrorMessage(msg, "CIOMap::CheckHessianNetworks");
      }
    }
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
    bool NetworkInference(const IOMap_Network &mapped_network) const { 
      bool inside = mapped_network.MLP->CheckInputInclusion();
      if (inside){
        mapped_network.MLP->CalcJacobian(mapped_network.evaluate_Jacobian);
        mapped_network.MLP->CalcHessian(mapped_network.evaluate_Hessian);
        mapped_network.MLP->Predict();
        RetrieveNetworkOutput(mapped_network);
        mapped_network.MLP->CalcJacobian(false);
        mapped_network.MLP->CalcHessian(false);
      }
      return inside;
    }

    /*!
    * \brief Check whether the network inputs and output are in the query
    * \param[in] network_to_check - pointer to network object.
    * \returns - if network input variables are in the query input and if at least one network output variable is in the query.
    */
    bool CheckNetworkVariables(const CNeuralNetwork  *network_to_check) {
        std::vector<std::string> network_inputs = network_to_check->GetInputVars();
        const std::vector<std::string> network_outputs = network_to_check->GetOutputVars();
        bool network_compatible{false};
        /* Check whether network all input variables are contained in query input. */
        for (auto q_in : query_input) {
            auto a = std::find(network_inputs.begin(), network_inputs.end(), q_in.first);
            if (a != std::end(network_inputs)) network_inputs.erase(a);
        }
        if (network_inputs.empty()) network_compatible = true;

        if (network_compatible) {
            /* Check if at least one network output is contained in the query output*/
            bool found_output{false};
            for (std::string var_out : network_outputs) {
                auto loc = std::find_if(query_output.begin(), query_output.end(), [var_out](std::pair<std::string, mlpdouble*>q) {return q.first==var_out;});
                if (loc != query_output.end()) found_output = true;
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
    
    /*!
    * \brief Check compatibility of query variables.
    */
    void CompatibilityChecks() const {
      /* Check if query inputs and outputs are unique sets. */
      CheckUniqueQueryVars(true);
      CheckUniqueQueryVars(false);  

      /* Check if no variables are shared between query input and output. */
      CheckSharedVars();

      /* Check if Jacobian enumerator and denominator are in query output and input respectively. */
      CheckJacobianQuery();

      /* Check if Hessian enumerator and denominators are in query output and input respectively. */
      CheckHessianQuery();
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
        /* Check if query input and output variables are compatible. */
        CompatibilityChecks();

        /* Collect query input and output variables without null */
        std::vector<std::string> query_vars_out = {}, query_vars_in = {};
        bool null_in_query{false};
        for (auto q_in : query_input) query_vars_in.push_back(q_in.first);
        for (auto q_out : query_output) {
          if (!CheckNull(q_out.first))
            query_vars_out.push_back(q_out.first);
          else null_in_query = true;
        }
        
        /* Copy of query variables used to check if all variables are covered by the networks. */
        auto remaining_query_vars_in = query_vars_in;
        auto remaining_query_vars_out = query_vars_out;

        /* Check network compatibility with query */
        query_network_maps.clear();
        for (auto network_to_check : networks_to_check) {
          bool compatible_input{false},
               compatible_output{false};
          std::vector<std::string> network_input_vars = network_to_check->GetInputVars();
          /* Check if the set of network input variables is a sub-set of the set of the query input variables */
          for (auto q_in : query_vars_in) {
            auto f = std::find(network_input_vars.begin(), network_input_vars.end(), q_in);
            if (f != network_input_vars.end()){
              network_input_vars.erase(f);
            }
          }
          if (network_input_vars.empty())
            compatible_input = true;
          
          if (compatible_input) {
            /* Check if any of the query output variables are in the set of network output variables . */
            compatible_output = false;
            std::vector<std::string> network_output_vars = network_to_check->GetOutputVars();
            for (auto q_out : query_vars_out) {
              auto f = std::find(network_output_vars.begin(), network_output_vars.end(), q_out);
              if (f != network_output_vars.end()){
                compatible_output = true;
              }
            }
          }

          if (compatible_input && compatible_output) {
            /* Add network to query */
            IOMap_Network mapped_network;
            mapped_network.MLP = network_to_check;
            query_network_maps.push_back(mapped_network);

            /* Update remaining query variables based on the network input and output variables. */
            const auto M_input = network_to_check->GetInputVars();
            for (auto m_in : M_input) {
              auto q = std::find(remaining_query_vars_in.begin(), remaining_query_vars_in.end(), m_in);
              if (q != remaining_query_vars_in.end())
                remaining_query_vars_in.erase(q);
            }
            const auto M_output = network_to_check->GetOutputVars();
            for (auto m_out : M_output) {
                auto q = std::find(remaining_query_vars_out.begin(), remaining_query_vars_out.end(), m_out);
                if (q != remaining_query_vars_out.end())
                  remaining_query_vars_out.erase(q);
              }
            }
          }
        
        /* Exit with an error if any of the query input and output variables are not included in the 
        network input and output sets. */
        if (!query_output.empty()) {
          if (!remaining_query_vars_out.empty()) {
            std::string msg = "The following query output variables are not present in the network output variables: ";
            for (auto v_out : remaining_query_vars_out)
              msg += (v_out + " ");
            ErrorMessage(msg, "CIOMap::FindNetworksForQuery");
          }
          if (!remaining_query_vars_in.empty() && !null_in_query) {
            std::string msg = "The following query input variables are not present in the network input variables: ";
            for (auto v_out : remaining_query_vars_in)
              msg += (v_out + " ");
            ErrorMessage(msg, "CIOMap::FindNetworksForQuery");
          }
        }

        /* Map network inputs and outputs to query inputs and outputs. */
        MapInputs();
        MapOutputs();

        /* Check if Jacobian queries are supported by compatible networks. */
        CheckJacobianNetworks();
        MapJacobians();

        /* Check if Hessian queries are supported by compatible networks. */
        CheckHessianNetworks();
        MapHessians();
    }

    /*!
    * \brief Set the input and evaluate the output of the mapped networks.
    */
    bool operator()() const 
    {
      bool within_bounds{true};
      for (const auto &mapped_network : query_network_maps) {
        SetNetworkInputs(mapped_network);
        if (!NetworkInference(mapped_network)) within_bounds=false;
      }
      SetNullOutputs();
      return within_bounds;
    };

    /*!
    * \brief Set the input and evaluate the output of the mapped networks.
    * \param[in] vals_input - query input values.
    * \param[in] refs_output - pointers to query output.
    */
    bool operator()(const std::vector<mlpdouble> &vals_input,const std::vector<mlpdouble*> &refs_output) const 
    {
      if (refs_output.size() != query_output_vals.size()){
        ErrorMessage("Number of outputs in query differs from number of requested outputs.", "CIOMap:operator()");
      }
      bool within_bounds{true};
      for (const auto &mapped_network : query_network_maps) {
        SetNetworkInputs(mapped_network, vals_input);
        if (!NetworkInference(mapped_network)) within_bounds=true;
      }
      SetNullOutputs();

      for (auto iOutput=0u; iOutput<query_output_vals.size(); iOutput++)
        *refs_output[iOutput] = query_output_vals[iOutput];
      return within_bounds;
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

    /*!
    * \brief Return the mean values of the input scaling parameters of the query networks.
    * \param[in] varname - name of the input variable for which to calculate the scaling parameters.
    * \returns - pair of values used for linear scaling. 
    */
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


