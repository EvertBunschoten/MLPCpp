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
#include "CNeuralNetwork.hpp"
namespace MLPToolbox {
class CIOMap
/*!
 *\class CIOMap
 *\brief This class is used by the CLookUp_ANN class to assign user-defined
 *inputs and outputs to loaded multi-layer perceptrons. When a look-up operation
 *is called with a specific CIOMap, the multi-layer perceptrons are evaluated
 *with input and output variables coinciding with the desired input and output
 *variable names.
 *
 *
 * For example, in a custom, data-driven fluid model, MLP's are used for
 *thermodynamic state definition. There are three MLP's loaded. MLP_1 predicts
 *temperature and specific heat based on density and energy. MLP_2 predicts
 *pressure and speed of sound based on density and energy as well. MLP_3
 *predicts density and energy based on pressure and temperature. During a
 *certain look-up operation in the CFluidModel, temperature, speed of sound and
 *pressure are needed for a given density and energy. What the CIOMap does is to
 *point to MLP_1 for temperature evalutation, and to MLP_2 for pressure and
 *speed of sound evaluation. MLP_3 is not considered, as the respective inputs
 *and outputs don't match with the function call inputs and outputs.
 *
 *  call variables:      MLP inputs:                     MLP outputs: call
 *outputs:
 *
 *                        2--> energy --|            |--> temperature --> 1
 *                                      |--> MLP_1 --|
 *  1:density            1--> density --|            |--> c_p 1:temperature
 *  2:energy 2:speed of sound 1--> density --|            |--> pressure --> 3
 *3:pressure
 *                                      |--> MLP_2 --|
 *                        2--> energy --|            |--> speed of sound --> 2
 *
 *                           pressure --|            |--> density
 *                                      |--> MLP_3 --|
 *                        temperature --|            |--> energy
 *
 *
 * \author E.Bunschoten
 */
{
private:
  std::vector<std::string> inputVariables, /*!< Input variable names for the
                                              current input-output map. */
      outputVariables; /*!< Output variable names for the current input-output
                          map. */

  std::vector<std::size_t> MLP_indices; /*!< Loaded MLP index */
  std::vector<std::vector<std::pair<std::size_t, std::size_t>>>
      Input_Map,  /*!< Mapping of call variable inputs to matching MLP inputs */
      Output_Map; /*!< Mapping of call variable outputs to matching MLP outputs
                   */
public:
  /*!
   * \brief Initiate input-output map with user-defined input and output
   * variables. \param[in] inputVariables_in - Vector containing input variable
   * names. \param[in] outputVariables_in - Vector containing output variable
   * names.
   */
  CIOMap(std::vector<std::string> &inputVariables_in,
         std::vector<std::string> &outputVariables_in) {
    inputVariables.resize(inputVariables_in.size());
    for (auto iVar = 0u; iVar < inputVariables_in.size(); iVar++) {
      inputVariables[iVar] = inputVariables_in[iVar];
    }
    outputVariables.resize(outputVariables_in.size());
    for (auto iVar = 0u; iVar < outputVariables_in.size(); iVar++) {
      outputVariables[iVar] = outputVariables_in[iVar];
    }
  }

  /*!
   * \brief Insert MLP index with stored input and output variables.
   * \param[in] iMLP - Loaded MLP index.
   */
  void PushMLPIndex(std::size_t iMLP) { MLP_indices.push_back(iMLP); }

  /*!
   * \brief Insert MLP input translation vector.
   * \param[in] inputIndices - Vector containing input variable index
   * translation.
   */
  void PushInputIndices(std::vector<std::pair<size_t, size_t>> inputIndices) {
    Input_Map.push_back(inputIndices);
  }

  /*!
   * \brief Insert MLP output translation vector.
   * \param[in] outputIndices - Vector containing output variable index
   * translation.
   */
  void PushOutputIndices(std::vector<std::pair<size_t, size_t>> outputIndices) {
    Output_Map.push_back(outputIndices);
  }

  /*!
   * \brief Get input variables of current input-output map.
   * \return Vector of input variables.
   */
  std::vector<std::string> GetInputVars() { return inputVariables; }

  /*!
   * \brief Get output variables of current input-output map.
   * \return Vector of output variables.
   */
  std::vector<std::string> GetOutputVars() { return outputVariables; }

  /*!
   * \brief Get the number of MLPs in the current IO map
   * \return number of MLPs with matching inputs and output(s)
   */
  std::size_t GetNMLPs() const { return MLP_indices.size(); }

  /*!
   * \brief Get the loaded MLP index
   * \return MLP index
   */
  std::size_t GetMLPIndex(std::size_t i_Map) const {
    return MLP_indices[i_Map];
  }

  /*!
   * \brief Get the call input variable index
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] iInput - input index of the call input variable
   * \return MLP input variable index
   */
  std::size_t GetInputIndex(std::size_t i_Map, std::size_t iInput) const {
    return Input_Map[i_Map][iInput].first;
  }

  /*!
   * \brief Get the call output variable index
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] iOutput - output index of the call input variable
   * \return call variable output index
   */
  std::size_t GetOutputIndex(std::size_t i_Map, std::size_t iOutput) const {
    return Output_Map[i_Map][iOutput].first;
  }

  /*!
   * \brief Get the MLP output variable index
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] iOutput - output index of the call input variable
   * \return MLP output variable index
   */
  std::size_t GetMLPOutputIndex(std::size_t i_Map, std::size_t iOutput) const {
    return Output_Map[i_Map][iOutput].second;
  }

  /*!
   * \brief Get the number of matching output variables between the call and MLP
   * outputs \param[in] i_Map - input-output mapping index of the IO map \return
   * Number of matching variables between the loaded MLP and call variables
   */
  std::size_t GetNMappedOutputs(std::size_t i_Map) const {
    return Output_Map[i_Map].size();
  }

  /*!
   * \brief Get the mapping of MLP outputs matching to call outputs
   * \param[in] i_Map - input-output mapping index of the IO map
   * \return Mapping of MLP output variables to call variables
   */
  std::vector<std::pair<std::size_t, std::size_t>>
  GetOutputMapping(std::size_t i_map) const {
    return Output_Map[i_map];
  }

  /*!
   * \brief Get the mapping of MLP inputs to call inputs
   * \param[in] i_Map - input-output mapping index of the IO map
   * \return Mapping of MLP input variables to call inputs
   */
  std::vector<std::pair<std::size_t, std::size_t>>
  GetInputMapping(std::size_t i_map) const {
    return Input_Map[i_map];
  }

  /*!
   * \brief Get the mapped inputs for the MLP at i_Map
   * \param[in] i_Map - input-output mapping index of the IO map
   * \param[in] inputs - call inputs
   * \return std::vector with call inputs in the correct order of the loaded MLP
   */
  std::vector<mlpdouble> GetMLPInputs(std::size_t i_Map,
                                      const std::vector<mlpdouble> &inputs) const {
    std::vector<mlpdouble> MLP_input;
    MLP_input.resize(Input_Map[i_Map].size());

    for (std::size_t iInput = 0; iInput < Input_Map[i_Map].size(); iInput++) {
      MLP_input[iInput] = inputs[GetInputIndex(i_Map, iInput)];
    }
    return MLP_input;
  }
};


class CQuery {

    std::vector<std::string> varnames_input;
    std::vector<std::string> varnames_output;
    std::vector<std::pair<std::string, mlpdouble*>> query_input;
    std::vector<std::pair<std::string, mlpdouble*>> query_output;
    
    bool evaluate_Jacobian {false};
    bool evaluate_Hessian {false};
    std::vector<std::pair<std::pair<std::string, std::string>, mlpdouble*>> query_Jacobian;
    std::vector<std::pair<std::pair<std::string, std::pair<std::string, std::string>>, mlpdouble*>> query_Hessian;

    std::vector<IteratorNetwork*> query_networks;
    
    std::vector<std::vector<std::pair<size_t, size_t>>> inputs_mapping;
    std::vector<std::vector<std::pair<size_t, size_t>>> outputs_mapping;
    std::vector<bool> query_networks_Jacobian;
    std::vector<bool> query_networks_Hessian;
    
    std::vector<std::vector<std::pair<std::pair<size_t, size_t>, mlpdouble*>>> Jacobians_mapping;
    std::vector<std::vector<std::pair<std::pair<size_t, std::pair<size_t, size_t>>, mlpdouble*>>> Hessians_mapping;
    public:
    void PushQueryVariable(std::string var_in) {
        varnames_input.push_back(var_in);
        
    }
    void PushSearchVariable(std::string var_out) {
        varnames_output.push_back(var_out);
    }

    void SetQueryInput(std::vector<std::pair<std::string, mlpdouble*>> const &q_input) {
        query_input.resize(q_input.size());
        varnames_input.resize(q_input.size());
        for (size_t iInput=0; iInput < q_input.size(); iInput++) {
            query_input[iInput] = std::make_pair(q_input[iInput].first, q_input[iInput].second);
            varnames_input[iInput] = q_input[iInput].first;
        }
    }

    void AddQueryInput(std::string varname, mlpdouble* ref_input) {
      query_input.push_back(std::make_pair(varname, ref_input));
    }

    void SetQueryOutput(std::vector<std::pair<std::string, mlpdouble*>> const &q_output) {
        query_output.resize(q_output.size());
        varnames_output.resize(q_output.size());
        for (size_t iOutput=0; iOutput < q_output.size(); iOutput++) {
            query_output[iOutput] = std::make_pair(q_output[iOutput].first, q_output[iOutput].second);
            varnames_output[iOutput] = q_output[iOutput].first;
        }
    }
    
    void AddQueryOutput(std::string varname, mlpdouble* ref_output) {
      query_output.push_back(std::make_pair(varname, ref_output));
    }

    void AddQueryJacobian(std::string varname_output, std::string varname_input, mlpdouble*ref_output) {
      evaluate_Jacobian = true;
      query_Jacobian.push_back(std::make_pair(std::make_pair(varname_output, varname_input),ref_output));
    }

    void AddQueryHessian(std::string varname_output, std::string varname_input_1,std::string varname_input_2, mlpdouble*ref_output) {
      evaluate_Hessian = true;
      evaluate_Jacobian = true;
      query_Hessian.push_back(std::make_pair(std::make_pair(varname_output, std::make_pair(varname_input_1, varname_input_2)),ref_output));
    }

    void SetQueryJacobian(std::vector<std::pair<std::pair<std::string, std::string>, mlpdouble*>> &Jacobian_input) {
        query_Jacobian = Jacobian_input;
    }

    void SetQueryHessian(std::vector<std::pair<std::pair<std::string, std::pair<std::string, std::string>>, mlpdouble*>> &Hessian_input) {
        query_Hessian = Hessian_input;
    }
    
    bool CheckNetworkVariables(IteratorNetwork * network_to_check) {
        std::vector<std::string> network_inputs = network_to_check->GetInputVars();
        std::vector<std::string> network_outputs = network_to_check->GetOutputVars();
        bool network_compatible{true};
        for (auto q_in : query_input) {
            auto a = find(network_inputs.begin(), network_inputs.end(), q_in.first);
            if (a == end(network_inputs)){
                network_compatible = false;
            }
        }
        if (network_compatible) {
            // for (std::string var_out : varnames_output) {
            //     auto loc = find(network_outputs.begin(), network_outputs.end(), var_out);
            //     if (loc == network_outputs.end())
            //         network_compatible = false;
            // }
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

    void FindNetworksForQuery(std::vector<IteratorNetwork*> &networks_to_check) {
        for (auto network_to_check : networks_to_check) {
            if (CheckNetworkVariables(network_to_check)){
                query_networks.push_back(network_to_check);
                query_networks_Hessian.push_back(false);
                query_networks_Jacobian.push_back(false);
            }
        }
        // if (!CheckUseOfOutputs()) {
        //     throw InsufficientOutputs();
        // }

        MapInputs();
        MapOutputs();
        if (evaluate_Jacobian) {
          MapJacobians();
          if (evaluate_Hessian) {
            MapHessians();
          }
        }

        SetQueryRefs();
    }

    void operator()() 
    {
      // for (IteratorNetwork *MLP : query_networks) {

      //   MLP->Predict();
      // }

      for (auto iNetwork=0u; iNetwork<query_networks.size(); iNetwork++) {
          query_networks[iNetwork]->CalcJacobian(query_networks_Jacobian[iNetwork]);
          query_networks[iNetwork]->CalcHessian(query_networks_Hessian[iNetwork]);
          //std::cout << query_networks_Jacobian[iNetwork] << std::endl;
          query_networks[iNetwork]->Predict();
          query_networks[iNetwork]->CalcJacobian(false);
          query_networks[iNetwork]->CalcHessian(false);
        }
    };

    bool CheckUseOfOutputs() {
        bool found_all_query_vars {true};
        for (std::string var_out : varnames_output) {
            bool found_var{false};
            for (auto network_to_check : query_networks) {
                auto vars_network_out = network_to_check->GetOutputVars();
                auto loc = find(vars_network_out.begin(), vars_network_out.end(), var_out);
                if (loc != vars_network_out.end()) {
                    found_var = true;
                }
            }
            if (!found_var){
                std::cout << "Warning! " << var_out << " is not present in the outputs of any of the loaded networks" << std::endl;
                found_all_query_vars = false;
            }
        }
        return found_all_query_vars;
    }

    void DisplayQueryNetworks() {
        for (auto q : query_networks) {
            q->DisplayNetwork();
            std::cout << std::endl;
        }
        
    }

    void MapInputs(){
        for (auto query_network : query_networks) {
            auto network_input_vars = query_network->GetInputVars();
            std::vector<std::pair<size_t, size_t>> input_map;
            for (size_t i_query=0; i_query<query_input.size(); i_query++) {
              for (size_t iInput=0; iInput < network_input_vars.size(); iInput++) {
                if (network_input_vars[iInput] == query_input[i_query].first)
                  input_map.push_back(std::make_pair(i_query, iInput));
              }
            }
            // for (size_t i_query=0; i_query < varnames_input.size(); i_query++) {
            //     for (size_t iInput=0; iInput < network_input_vars.size(); iInput++) {
            //         if (network_input_vars[iInput] == varnames_input[i_query]) {
            //             input_map.push_back(std::make_pair(i_query, iInput));
            //         }
            //     }
            // }
            inputs_mapping.push_back(input_map);
        }
    }

    void MapOutputs(){
        for (auto query_network : query_networks) {
            auto network_output_vars = query_network->GetOutputVars();
            std::vector<std::pair<size_t, size_t>> output_map;
            for (size_t i_query=0; i_query<query_output.size(); i_query++) {
              for (size_t iOutput=0; iOutput < network_output_vars.size(); iOutput++) {
                if (network_output_vars[iOutput] == query_output[i_query].first)
                  output_map.push_back(std::make_pair(i_query, iOutput));
              }
            }
            outputs_mapping.push_back(output_map);
        }
    }

    void MapJacobians() {
      for (size_t iNetwork=0; iNetwork<query_networks.size(); iNetwork++) {
        auto query_network = query_networks[iNetwork];
        auto network_output_vars = query_network->GetOutputVars();
        auto network_input_vars = query_network->GetInputVars();
        for (size_t i_query=0; i_query<query_Jacobian.size(); i_query++) {
          std::vector<std::pair<std::pair<size_t, size_t>, mlpdouble*>> Jacobian_map;
          for (size_t iOutput=0; iOutput < network_output_vars.size(); iOutput++) {
            if (network_output_vars[iOutput] == query_Jacobian[i_query].first.first){
              for (size_t iInput=0; iInput < network_input_vars.size(); iInput++) {
                if (network_input_vars[iInput] == query_Jacobian[i_query].first.second){
                  auto Jacobian_map_network = std::make_pair(iOutput, iInput);
                  auto query_binding = std::make_pair(Jacobian_map_network, query_Jacobian[i_query].second);
                  Jacobian_map.push_back(query_binding);
                }
              }
            }
          }
          Jacobians_mapping.push_back(Jacobian_map);
          query_networks_Jacobian[iNetwork] = true;
        }
      }
    }

    void MapHessians() {
      for (size_t iNetwork=0; iNetwork<query_networks.size(); iNetwork++) {
        auto query_network = query_networks[iNetwork];
        auto network_output_vars = query_network->GetOutputVars();
        auto network_input_vars = query_network->GetInputVars();
        for (size_t i_query=0; i_query<query_Hessian.size(); i_query++) {
          std::vector<std::pair<std::pair<size_t, std::pair<size_t, size_t>>, mlpdouble*>> Hessian_map;
          for (size_t iOutput=0; iOutput < network_output_vars.size(); iOutput++) {
            if (network_output_vars[iOutput] == query_Hessian[i_query].first.first){
              for (size_t iInput=0; iInput < network_input_vars.size(); iInput++) {
                if (network_input_vars[iInput] == query_Hessian[i_query].first.second.first){
                  for (size_t jInput=0; jInput < network_input_vars.size(); jInput++) {
                    if (network_input_vars[jInput] == query_Hessian[i_query].first.second.second){
                      auto Hessian_map_network = std::make_pair(iOutput, std::make_pair(iInput, jInput));
                      auto query_binding = std::make_pair(Hessian_map_network, query_Hessian[i_query].second);
                      Hessian_map.push_back(query_binding);
                    }
                  }
                }
              }
            }
          }
          Hessians_mapping.push_back(Hessian_map);
          query_networks_Hessian[iNetwork] = true;
        }
      }
    }
    void SetQueryRefs() {
        for(size_t iNetwork=0; iNetwork < query_networks.size(); iNetwork++) {
            
            for (size_t iInput=0; iInput < query_input.size(); iInput++) {
                size_t jInput = inputs_mapping[iNetwork][iInput].second;
                query_networks[iNetwork]->SetInputRef(inputs_mapping[iNetwork][iInput].second, query_input[inputs_mapping[iNetwork][iInput].first].second);
            }
            for (size_t iOutput=0; iOutput < outputs_mapping[iNetwork].size(); iOutput++) {
                query_networks[iNetwork]->SetOutputRef(outputs_mapping[iNetwork][iOutput].first, query_output[outputs_mapping[iNetwork][iOutput].second].second);
                
                // if (evaluate_Jacobian) {
                //     for (size_t iInput=0; iInput < varnames_input.size(); iInput++)
                //         query_networks[iNetwork]->SetJacobianRef(outputs_mapping[iNetwork][iOutput].first, inputs_mapping[iNetwork][iInput].second, query_Jacobian[iOutput][inputs_mapping[iNetwork][iInput].second]);
                // }
            }
            if (evaluate_Jacobian) {
              for (size_t iJacobian=0; iJacobian < Jacobians_mapping.size(); iJacobian++) {

                //query_networks[iNetwork]->CalcJacobian(true);
                query_networks[iNetwork]->SetJacobianRef(Jacobians_mapping[iJacobian][iNetwork].first.first, Jacobians_mapping[iJacobian][iNetwork].first.second, Jacobians_mapping[iJacobian][iNetwork].second);
              }
            
            }
            if (evaluate_Hessian) {
              for (size_t iHessian=0; iHessian<Hessians_mapping.size(); iHessian++) {
                //query_networks[iNetwork]->CalcHessian(true);
                query_networks[iNetwork]->SetHessianRef(Hessians_mapping[iHessian][iNetwork].first.first, Hessians_mapping[iHessian][iNetwork].first.second.first, Hessians_mapping[iHessian][iNetwork].first.second.second, Hessians_mapping[iHessian][iNetwork].second);
              }
            }

            
        }
    }

};
} // namespace MLPToolbox


