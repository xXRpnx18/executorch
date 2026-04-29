/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/training/module/training_module.h>

#include <string>

namespace executorch {
namespace extension {
namespace training {

namespace {

std::string make_parameters_method_name(const std::string& method_name) {
  return "__et_training_parameters_index_" + method_name;
}

std::string make_gradients_method_name(const std::string& method_name) {
  return "__et_training_gradients_index_" + method_name;
}

std::string make_fqn_method_name(const std::string& method_name) {
  return "__et_training_fqn_" + method_name;
}

} // namespace

runtime::Result<std::vector<runtime::EValue>>
TrainingModule::execute_forward_backward(
    const std::string& method_name,
    const std::vector<runtime::EValue>& input) {
  // Find where the user outputs end.
  const std::string gradients_method_name =
      make_gradients_method_name(method_name);
  auto res = executorch::extension::Module::execute(gradients_method_name);
  if (!res.ok()) {
    return res.error();
  }
  uint64_t grad_start = res.get()[0].toInt();

  const std::string parameters_method_name =
      make_parameters_method_name(method_name);
  // get params start.
  auto param_res =
      executorch::extension::Module::execute(parameters_method_name);
  if (!param_res.ok()) {
    return param_res.error();
  }

  uint64_t param_start = param_res.get()[0].toInt();

  // Execute the forward and backward pass.
  auto outputs = torch::executor::Module::execute(method_name, input);
  if (!outputs.ok()) {
    return outputs.error();
  }

  // Extract the user outputs.
  std::vector<runtime::EValue> user_outputs;
  user_outputs.reserve(grad_start);
  for (size_t i = 0; i < grad_start; ++i) {
    user_outputs.push_back(outputs.get().at(i));
  }

  // Extract and store the gradients and params if this is the first time seeing
  // this method.
  if (method_named_gradients_.find(method_name) ==
      method_named_gradients_.end()) {
    // Fully qualified names
    std::vector<runtime::EValue> fqn_list;
    method_named_gradients_.insert({method_name, {}});

    auto& gradients_map = method_named_gradients_.at(method_name);

    // Get names if we havent seen this method before.
    const std::string fqn_method_name = make_fqn_method_name(method_name);
    auto fqn_res = executorch::extension::Module::execute(fqn_method_name);
    if (!fqn_res.ok()) {
      return fqn_res.error();
    }
    fqn_list = fqn_res.get();

    // Map every gradient output to its state_dict FQN (target) so optimizers and
    // int8 buffer updates can find ∂L/∂w and ∂L/∂b.
    //
    //  (1) |fqn| == n_grad: one FQN per slot (include GRADIENT_TO_USER_INPUT in emit).
    //  (2) YOLO11 QAT joint (edge int8): n_grad == 48 and __et_training_fqn lists 24 bias
    //      names only — (weight, bias)×24 with `.bias`→`.weight` for the weight keys.
    //
    // 47-slot legacy PTE and generic prefix/suffix heuristics are **not** supported;
    // re-export with a current ExecuTorch emit (`_get_training_metadata` aligned with
    // visible outputs).
    const size_t n_grad_slots = static_cast<size_t>(param_start - grad_start);
    if (fqn_list.size() == n_grad_slots) {
      for (size_t j = 0; j < n_grad_slots; j++) {
        const size_t grad_index = static_cast<size_t>(grad_start) + j;
        std::string_view fqn = fqn_list.at(j).toString();
        gradients_map.insert({fqn, outputs.get().at(grad_index).toTensor()});
      }
    } else if (fqn_list.size() == 24 && n_grad_slots == 48) {
      // YOLO11 QAT PTE (PyTorch 2.8 joint, edge int8): 48 gradient slots are
      // (weight, bias) × 24.  Some emits list only the 24 bias FQNs in pair order
      // (same stem order as the legacy 47-slot case, but all 24 head biases).
      for (size_t j = 0; j < 24; j++) {
        std::string bname(std::string(fqn_list.at(j).toString()));
        std::string wname = bname;
        const auto pos = wname.rfind(".bias");
        if (pos != std::string::npos) {
          wname.replace(pos, 5, ".weight");
        }
        const size_t g_w = static_cast<size_t>(grad_start) + 2 * j;
        const size_t g_b = g_w + 1;
        gradients_map.insert(
            {std::string_view(wname), outputs.get().at(g_w).toTensor()});
        gradients_map.insert(
            {std::string_view(bname), outputs.get().at(g_b).toTensor()});
      }
      ET_LOG(
          Info,
          "execute_forward_backward: YOLO11 layer-23 interleaved (w,b)×24, fqns=24 slots=%zu",
          n_grad_slots);
    } else {
      ET_LOG(
          Error,
          "execute_forward_backward: unsupported grad/FQN layout (slots=%zu fqn_count=%zu). "
          "YOLO11 QAT joint requires 48 gradient slots with either |fqn|==48 or "
          "(|fqn|==24 && slots==48). Re-export the .pte; 47-slot legacy is removed.",
          n_grad_slots,
          fqn_list.size());
      return executorch::runtime::Error::InvalidArgument;
    }
  }

  return user_outputs;
}

runtime::Result<const std::map<std::string_view, executorch::aten::Tensor>>
TrainingModule::named_parameters(const std::string& method_name) {
  // If we haven't seen this method before, populate the dict.
  if (method_named_parameters_.find(method_name) ==
      method_named_parameters_.end()) {
    const std::string fqn_method_name = make_fqn_method_name(method_name);
    const std::string parameters_method_name =
        make_parameters_method_name(method_name);

    method_named_parameters_.insert({method_name, {}});

    // get names.
    auto fqn_res = executorch::extension::Module::execute(fqn_method_name);
    if (!fqn_res.ok()) {
      return fqn_res.error();
    }
    const auto& fqn_list = fqn_res.get();

    // get params start.
    auto param_res =
        executorch::extension::Module::execute(parameters_method_name);
    if (!param_res.ok()) {
      return param_res.error();
    }

    uint64_t param_start = param_res.get()[0].toInt();

    // Gradient slot count (for FQN ordering: some exports interleave
    // [grad0..gradN, param0..paramM] in a single fqn list).
    const std::string grad_method = make_gradients_method_name(method_name);
    auto grad_res = executorch::extension::Module::execute(grad_method);
    if (!grad_res.ok()) {
      return grad_res.error();
    }
    const uint64_t grad_start = grad_res.get()[0].toInt();

    // Load the method if it is not already loaded.
    auto e = executorch::extension::Module::load_method(method_name);
    if (e != runtime::Error::Ok) {
      return e;
    }
    auto& method = methods_.at(method_name).method;

    const size_t n_param_out = method->outputs_size() - static_cast<size_t>(param_start);
    const size_t n_grad = static_cast<size_t>(param_start - grad_start);
    // Match export layout: either fqn = [all grads, all params] or fqn = params only.
    size_t fqn_base = 0;
    if (fqn_list.size() == n_grad + n_param_out) {
      fqn_base = n_grad;
    } else if (fqn_list.size() < n_param_out) {
      ET_LOG(
          Info,
          "named_parameters: fqn list (%zu) < param outputs (%zu); will bind min count",
          fqn_list.size(),
          n_param_out);
    }

    for (size_t i = 0; i < n_param_out; ++i) {
      const size_t fqn_i = fqn_base + i;
      if (fqn_i >= fqn_list.size()) {
        ET_LOG(
            Info,
            "named_parameters: extra output at index %zu with no fqn; stopping at %zu params",
            static_cast<size_t>(param_start) + i,
            i);
        break;
      }
      const size_t param_index = static_cast<size_t>(param_start) + i;
      std::string_view fqn = fqn_list.at(fqn_i).toString();
      executorch::aten::Tensor param =
          method->get_output(param_index).toTensor();
      method_named_parameters_.at(method_name).insert({fqn, param});
    }
  }
  return method_named_parameters_.at(method_name);
}

runtime::Result<const std::map<std::string_view, executorch::aten::Tensor>>
TrainingModule::named_gradients(const std::string& method_name) {
  if (method_named_gradients_.find(method_name) ==
      method_named_gradients_.end()) {
    ET_LOG(Error, "No gradients found for method %s", method_name.c_str());
    return executorch::runtime::Error::InvalidArgument;
  }
  return method_named_gradients_.at(method_name);
}

runtime::Result<const std::map<std::string_view, executorch::aten::Tensor>>
TrainingModule::named_attributes(const std::string& method_name) {
  // If we haven't seen this method before, populate the dict.
  if (method_named_attributes_.find(method_name) ==
      method_named_attributes_.end()) {
    method_named_attributes_.insert({method_name, {}});

    // get method metadata
    auto meta_res = method_meta(method_name);
    if (!meta_res.ok()) {
      return meta_res.error();
    }
    // get method
    auto e = load_method(method_name);
    if (e != runtime::Error::Ok) {
      return e;
    }
    auto& method = methods_.at(method_name).method;
    // get tensor by name
    for (int idx = 0; idx < meta_res->num_attributes(); idx++) {
      const auto tensor_res = meta_res->attribute_tensor_meta(idx);
      if (!tensor_res.ok()) {
        return tensor_res.error();
      }
      const auto tensorName = tensor_res.get().name();
      const auto attribute_res = method->get_attribute(tensorName);
      if (!attribute_res.ok()) {
        return attribute_res.error();
      }
      method_named_attributes_.at(method_name)
          .insert({tensorName, attribute_res.get()});
    }
  }
  return method_named_attributes_.at(method_name);
}

} // namespace training
} // namespace extension
} // namespace executorch
