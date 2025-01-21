# Hasktorch Compose

In hasktorch, model specifications, values, and inference are defined separately. there are cases combining commonly used models. For example, we might want to connect three linear layers, or add one linear layer to an existing model.
This repo provides a library to easily compose existing models.
We plan to provide both an untyped API and a typed API, but we will prioritize the development of the untyped API.

This is an experimental library developed based on [hasktorch-skeleton](https://github.com/hasktorch/hasktorch-skeleton).

List of planned features:

- [x] Sequential
- [ ] Extract layer
- [ ] Test for each layer
- [ ] Overlay layer
- [ ] Concatenate layer

# Examples

## Sequential

This is an example of an MLP implementation, created by combining LinearSpec.

```haskell
type MLPSpec = LinearSpec :>>: ReluSpec :>>: LinearSpec :>>: ReluSpec :>>: LinearSpec
type MLP = Linear :>>: (Relu :>>: (Linear :>>: (Relu :>>: Linear)))

mlpSpec =
  Forward (LinearSpec 784 64) $
  Forward ReluSpec $
  Forward (LinearSpec 64 32) $
  Forward ReluSpec $
  LinearSpec 32 10

mlp :: (Randomizable MLPSpec MLP, HasForward MLP Tensor Tensor) => MLP -> Tensor -> Tensor
mlp model input =
  logSoftmax (Dim 1) $ forward model input
```

## Extract layer

## Test for each layer

For one input, take the outputs of all layers, then compare the shapes and values of all the layers.

