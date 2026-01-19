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
        for (const auto var : input_names) message += (var + ", ");

        char *cstr = new char[message.size() + 1];
        std::strcpy(cstr, message.c_str());
        return cstr;
      }
  };

  
struct IOMap_Network {
  IteratorNetwork* MLP;
  std::vector<std::pair<mlpdouble*, mlpdouble*>> input_map;
  std::vector<std::pair<mlpdouble*,mlpdouble*>> output_map;
  std::vector<std::pair<const mlpdouble*, const mlpdouble*>> Jacobian_map;
  std::vector<std::pair<const mlpdouble*, const mlpdouble*>> Hessian_map;
  bool evaluate_Jacobian {false};
  bool evaluate_Hessian {false};
};

class CQuery {

    std::vector<std::pair<std::string, mlpdouble*>> query_input;
    std::vector<std::pair<std::string, mlpdouble*>> query_output;
    std::vector<mlpdouble> mean_query_inputs_min,
                           mean_query_inputs_max,
                           mean_query_inputs_offset;
    
    std::vector<IOMap_Network> query_network_maps;

    std::vector<std::pair<std::pair<std::string, std::string>, mlpdouble*>> query_Jacobian;
    std::vector<std::pair<std::pair<std::string, std::pair<std::string, std::string>>, mlpdouble*>> query_Hessian;

    std::vector<mlpdouble> query_output_vals;
    
    public:

    const std::vector<IOMap_Network> GetNetworksInQuery() const {return query_network_maps;}
    // const std::pair<mlpdouble, mlpdouble> GetMeanInputBounds() const {return std::make_pair(mean_input_bounds_min,mean_input_bounds_max);}
    // const mlpdouble GetMeanInputOffset() const {return std::make_pair(mean_input_bounds_min,mean_input_bounds_max);}
    
    void SetQueryInput(const std::vector<std::string> &varnames) {
      query_input.clear();
      for (auto var : varnames)
        query_input.push_back(std::make_pair(var, nullptr));
    }

    void SetQueryOutput(const std::vector<std::string> &varnames) {
      query_output.clear();
      query_output_vals.resize(varnames.size());
      for (auto iOutput=0u; iOutput<varnames.size(); iOutput++)
        query_output.push_back(std::make_pair(varnames[iOutput], &query_output_vals[iOutput]));
    }
    
    std::vector<mlpdouble> GetQueryOutput() const {return query_output_vals;}

    void AddQueryInput(const std::string varname, mlpdouble* ref_input=nullptr) {
      query_input.push_back(std::make_pair(varname, ref_input));
    }
    
    void AddQueryOutput(const std::string varname, mlpdouble* ref_output=nullptr) {
      query_output.push_back(std::make_pair(varname, ref_output));
    }

    void AddQueryJacobian(const std::string varname_output, const std::string varname_input, mlpdouble*ref_output) {
      query_Jacobian.push_back(std::make_pair(std::make_pair(varname_output, varname_input),ref_output));
    }

    void AddQueryHessian(const std::string varname_output, const std::string varname_input_1, const std::string varname_input_2, mlpdouble*ref_output) {
      query_Hessian.push_back(std::make_pair(std::make_pair(varname_output, std::make_pair(varname_input_1, varname_input_2)),ref_output));
    }

    bool CheckNetworkVariables(const IteratorNetwork  *network_to_check) {
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

    bool CheckUniqueInputs(const IteratorNetwork  *network_to_check) const {
      auto network_inputs = network_to_check->GetInputVars();
      std::sort(network_inputs.begin(),network_inputs.end());
      return std::unique(network_inputs.begin(),network_inputs.end()) == network_inputs.end();
    }

    void FindNetworksForQuery(const std::vector<IteratorNetwork*> &networks_to_check) {
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

    void operator()() 
    {
      for (const auto &mapped_network : query_network_maps) {
        SetNetworkInputs(mapped_network);
        if (mapped_network.MLP->CheckInputInclusion()){
          mapped_network.MLP->CalcJacobian(mapped_network.evaluate_Jacobian);
          mapped_network.MLP->CalcHessian(mapped_network.evaluate_Hessian);
          mapped_network.MLP->Predict();
          RetrieveNetworkOutput(mapped_network);
          mapped_network.MLP->CalcJacobian(false);
          mapped_network.MLP->CalcHessian(false);
        }
      }
    };

    void operator()(const std::vector<mlpdouble> &vals_input,const std::vector<mlpdouble*> &refs_output) 
    {
      if (refs_output.size() != query_output_vals.size()){
        throw std::exception();
      }
      // std::vector<mlpdouble> dist_vals;
      // for (const auto &mapped_network : query_network_maps) {
      //   SetNetworkInputs(mapped_network, vals_input);
      //   const auto val_dist = mapped_network.MLP->QueryDistance();
      //   dist_vals.push_back(val_dist);
      // }
      for (const auto &mapped_network : query_network_maps) {
        SetNetworkInputs(mapped_network, vals_input);
        if (mapped_network.MLP->CheckInputInclusion()){
          mapped_network.MLP->CalcJacobian(mapped_network.evaluate_Jacobian);
          mapped_network.MLP->CalcHessian(mapped_network.evaluate_Hessian);
          mapped_network.MLP->Predict();
          RetrieveNetworkOutput(mapped_network);
          mapped_network.MLP->CalcJacobian(false);
          mapped_network.MLP->CalcHessian(false);
        }
      }
      for (auto iOutput=0u; iOutput<query_output_vals.size(); iOutput++)
        *refs_output[iOutput] = query_output_vals[iOutput];
    };

    void SetNetworkInputs(const IOMap_Network &mapped_network) const {
      for (const auto &q_in : mapped_network.input_map) {
        *q_in.second = *q_in.first;
      }
    } 

    void SetNetworkInputs(const IOMap_Network &mapped_network, const std::vector<mlpdouble> &vals_input) const {
      for (auto iInput=0u; iInput < mapped_network.input_map.size(); iInput++) {
        *mapped_network.input_map[iInput].second = vals_input[iInput];
      }
    }

    void RetrieveNetworkOutput(const IOMap_Network &mapped_network) const {
      for (const auto &q_out : mapped_network.output_map) {
        *q_out.first = *q_out.second;
      }
    }

    // void RetrieveNetworkOutput(const IOMap_Network &mapped_network, const std::vector<mlpdouble*> &vals_output) const {

    //   for (auto iOutput=0u; iOutput < mapped_network.output_map.size(); iOutput++) {
    //     *vals_output[iOutput] = *mapped_network.output_map[iOutput].second;
    //   }
    // }

    bool CheckUseOfOutputs() const {
        bool found_all_query_vars {true};
        for (auto &q : query_output) {
          bool found_var{false};
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

    void DisplayQueryNetworks() const {
        for (const auto &mapped_network : query_network_maps) {
            mapped_network.MLP->DisplayNetwork();
            std::cout << std::endl;
        }
        
    }

    
    void MapInputs(){
        // mean_query_inputs_max.resize(query_input.size());
        // mean_query_inputs_min.resize(query_input.size());
        // mean_query_inputs_offset.resize(query_input.size());

        // for (auto iInput=0u; iInput<query_input.size(); iInput++) {
        //   mean_query_inputs_max[iInput]=0;
        //   mean_query_inputs_min[iInput]=0;
        //   mean_query_inputs_offset[iInput]=0;
        // }
        for (auto &mapped_network : query_network_maps) {
          const auto network_input_vars = mapped_network.MLP->GetInputVars();
          for (const auto &q : query_input) {

            auto loc=std::find(network_input_vars.begin(), network_input_vars.end(), q.first);
            if (loc != network_input_vars.end()) {
              auto ref_query_input = q.second;
              auto iInput_network = distance(network_input_vars.begin(), loc);
              auto ref_network_input = mapped_network.MLP->InputLayer(iInput_network);
              const auto input_norm = mapped_network.MLP->GetInputNorm(iInput_network);
              const mlpdouble input_offset = mapped_network.MLP->GetRegularizationOffset(iInput_network);
              const mlpdouble input_min = input_norm.first;
              const mlpdouble input_max = input_norm.second;
              mean_query_inputs_min.push_back(input_min / query_network_maps.size());

              // avg_input_min += input_min;
              // avg_input_max += input_max;
              // avg_input_offset += input_offset;
              mapped_network.input_map.push_back(std::make_pair(ref_query_input, ref_network_input));
            }
          }
        }
        // avg_input_min /= query_network_maps.size();
        // avg_input_max /= query_network_maps.size();
        // avg_input_offset /= query_network_maps.size();

        // mean_input_bounds_min = avg_input_min;
        // mean_input_bounds_max = avg_input_max;
        // mean_input_offset = avg_input_offset;
    }

    void MapOutputs(){
      for (auto &mapped_network : query_network_maps) {
        auto network_output_vars = mapped_network.MLP->GetOutputVars();
        for (const auto &q : query_output) {
          auto loc = std::find(network_output_vars.begin(), network_output_vars.end(), q.first);
          if (loc != network_output_vars.end()) {
            auto iOutput = distance(network_output_vars.begin(), loc);
            auto ref_query_output = q.second;
            auto ref_network_output = mapped_network.MLP->OutputLayer(iOutput);
            mapped_network.output_map.push_back(std::make_pair(ref_query_output, ref_network_output));          
          }
        }
      }
    }

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

    void MapHessians() {
      for (auto &mapped_network : query_network_maps) {
        const auto network_output_vars = mapped_network.MLP->GetOutputVars();
        const auto network_input_vars = mapped_network.MLP->GetInputVars();
        for (const auto &q : query_Hessian) {
          const std::string varname_enumerator = q.first.first;
          const std::string varname_demominator_1 = q.first.second.first;
          const std::string varname_demominator_2 = q.first.second.second;
          const auto ref_output_Hessian = q.second;

          auto loc_enumerator = std::find(network_output_vars.begin(), network_output_vars.end(), varname_enumerator);
          if (loc_enumerator != network_output_vars.end()) {
            const auto iOutput = distance(network_output_vars.begin(), loc_enumerator);
            auto loc_demoninator_1 = std::find(network_input_vars.begin(), network_input_vars.end(), varname_demominator_1);
            if (loc_demoninator_1 != network_input_vars.end()) {
              const auto iInput = distance(network_input_vars.begin(), loc_demoninator_1);
              auto loc_denominator_2 = std::find(network_input_vars.begin(), network_input_vars.end(), varname_demominator_2);
              if (loc_denominator_2 != network_input_vars.end()) {
                const auto jInput = distance(network_input_vars.begin(), loc_denominator_2);
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

};
} // namespace MLPToolbox


