#include <string>
#include <vector>
#include <iostream>
namespace MLPToolbox {
class QueryOutputNotFoundException : public std::exception
{
  public:
    QueryOutputNotFoundException() noexcept = default;
    ~QueryOutputNotFoundException() noexcept = default;
    virtual const char* what() const noexcept {
      return "Query output variables not present in network output variables.";
    }
};

class QueryInputNotFoundException : public std::exception
{
  public:
    QueryInputNotFoundException() noexcept = default;
    ~QueryInputNotFoundException() noexcept = default;
    virtual const char* what() const noexcept {
      return "Query input variables not present in network input variables.";
    }
};

class DuplicatesInQueryException : public std::exception 
{
  private:
    bool is_input;
    std::string duplicate_var;
  public:
    DuplicatesInQueryException(const bool in_input, const std::string &var) noexcept : is_input{in_input}, duplicate_var{var} {};
    ~DuplicatesInQueryException() noexcept = default;
    virtual const char* what() const noexcept {
      std::string msg = "Query contains duplicate ";
      if (is_input)
        msg += "input :";
      else
        msg += "output :";
      msg += duplicate_var;

      return msg.c_str();
    }
};

class SharedInputsOutputsException : public std::exception 
{
  private:
  std::string shared_var;
  public:
  SharedInputsOutputsException(const std::string &var) noexcept : shared_var{var} {};
  ~SharedInputsOutputsException() noexcept = default;
  virtual const char* what() const noexcept {
    std::string msg = "Query shares ";
    msg += shared_var;
    msg += " between inputs and outputs";
    return msg.c_str();
  }
};

class InsufficientJacobianException : public std::exception
{
  private:
  std::string missing_enumerators,
              missing_denominators;
  public:
  InsufficientJacobianException(const std::string & enumerators, const std::string & denominators) noexcept : missing_enumerators{enumerators}, missing_denominators{denominators} {};
  ~InsufficientJacobianException() noexcept = default;
  virtual const char* what() const noexcept {
    std::string msg = "Jacobian enumerator variables not included in query: ";
    msg += missing_enumerators;
    msg += " Jacobian denominator  variables not included in query: ";
    msg += missing_denominators;
    return msg.c_str();
  }
};

class InsufficientHessianException : public std::exception
{
  private:
  std::string missing_enumerators,
              missing_denominators;
  public:
  InsufficientHessianException(const std::string & enumerators, const std::string & denominators) noexcept : missing_enumerators{enumerators}, missing_denominators{denominators} {};
  ~InsufficientHessianException() noexcept = default;
  virtual const char* what() const noexcept {
    std::string msg = "Hessian enumerator variables not included in query: ";
    msg += missing_enumerators;
    msg += " Hessian denominator  variables not included in query: ";
    msg += missing_denominators;
    return msg.c_str();
  }
};

class JacobianNotSupportedException : public std::exception
{
  private:
  std::vector<std::string> impossible_jacobians;
  public:
  JacobianNotSupportedException(const std::vector<std::string> &jacs) noexcept : impossible_jacobians{jacs} {};
  ~JacobianNotSupportedException() noexcept = default;
  virtual const char* what() const noexcept {
    std::string msg = "The following Jacobian queries were not supported by the networks: ";
    for (auto j : impossible_jacobians) {
        msg += (j + " ");
    }
    return msg.c_str();
  }
};

class HessianNotSupportedException : public std::exception
{
  private:
  std::vector<std::string> impossible_hessians;
  public:
  HessianNotSupportedException(const std::vector<std::string> &hes) noexcept : impossible_hessians{hes} {};
  ~HessianNotSupportedException() noexcept = default;
  virtual const char* what() const noexcept {
    std::string msg = "The following Hessian queries were not supported by the networks: ";
    for (auto j : impossible_hessians) {
        msg += (j + " ");
    }
    return msg.c_str();
  }
};

class IncompatibleOutputVectorException : public std::exception
{
  
  public:
  IncompatibleOutputVectorException() noexcept = default;
  ~IncompatibleOutputVectorException() noexcept = default;
  virtual const char* what() const noexcept {
    std::string msg = "Number of outputs in query differs from number of requested outputs.";
    return msg.c_str();
  }
};
}