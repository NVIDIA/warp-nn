# Changelog

## [0.4.0] - Unreleased
### Added
- Add `Buffer` class and module buffers (non-learnable state arrays, such as running statistics)
- Add training/evaluation modes to modules (`training` property, and `train()` and `eval()` methods)
- Add the `requires_grad` constructor argument to modules that own parameters and/or cached output arrays,
  to control whether those arrays require gradients
- Add the `initialize_parameters` constructor argument to layers that own parameters,
  to control whether those parameters are initialized with their default/initial values
- Add `Abs`, `Acos`, `Acosh`, `Add`, `Asin`, `Asinh`, `Atan`, `Atanh`, `AvgPool1D`, `AvgPool2D`, `BatchNorm`,
  `BitwiseAnd`, `BitwiseNot`, `BitwiseOr`, `BitwiseXor`, `Ceil`, `CELU`, `Clip`, `Cos`, `Cosh`, `Div`, `Dropout`,
  `Erf`, `Exp`, `Floor`, `GELU`, `GlobalAvgPool`, `GlobalMaxPool`, `GroupNorm`, `HardSigmoid`, `HardSwish`,
  `Identity`, `InstanceNorm`, `LayerNorm`, `Log`, `LogSoftmax`, `Max`, `MaxPool1D`, `MaxPool2D`, `Min`, `Mish`, `Mul`,
  `Neg`, `Pow`, `PReLU`, `Reciprocal`, `RMSNorm`, `Round`, `Shrink`, `Sign`, `Sin`, `Sinh`, `Softmax`, `Sqrt`, `Sub`,
  `Swish`, `Tan` and `Threshold` modules
- Add `OnnxRuntimeV2` ONNX inference runtime and mark `OnnxRuntime` as deprecated

### Changed
- Update the minimum required Warp version to 1.15.0

### Fixed
- Fix crash when calling a module after moving it to another device (the cached arrays were not reallocated)
- Fix optimizers' CUDA graph capture and kernel-sharing bugs: first `step()`/`clip_by_total_norm()` call,
  gradient clipping, and per-optimizer `eps`/`max_norm` handling
- Fix numerical issues of activations: `ReLU` did not propagate NaN inputs, `Sigmoid` yielded NaN gradients
  for large negative inputs, and `Softplus` overflowed (with NaN gradients) for large positive inputs

## [0.3.1] - 2026-07-28
### Changed
- Update the ONNX inference runtime to include opt-in support for gradient propagation

## [0.3.0] - 2026-07-03
### Added
- Add graph-capturable `OnnxRuntime` ONNX inference runtime

## [0.2.0] - 2026-05-23
### Added
- Add `Flatten` layer

## [0.1.0] - 2026-04-08
### Added
- Initial release
