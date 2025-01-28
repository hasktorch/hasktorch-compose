# Hasktorch Compose

In Hasktorch, model specifications, values, and inference are defined separately. This often necessitates combining commonly used models. For example, three linear layers may need to be connected in sequence, or a new linear layer could be added to an existing model. Hasktorch Compose provides a straightforward way to compose such models.

In addition to simple model composition, this library aims to support extracting parts of models and sharing parameters between different models, such as ControlNet and RoLa. Both an untyped API and a typed API are planned, with initial development focused on the untyped API.

Hasktorch Compose is an experimental library built on top of [hasktorch-skeleton](https://github.com/hasktorch/hasktorch-skeleton).

**Planned Features:**
- [x] Sequential
- [ ] Extract layer
- [ ] Test for each layer
- [ ] Overlay layer
- [x] Concatenate layer

# Examples

## Sequential

Use `.*.*` operator of HList to join layers.
This is an example of an MLP implementation, created by combining LinearSpec.

```haskell
newtype MLPSpec = MLPSpec (HList [LinearSpec, ReluSpec, LinearSpec, ReluSpec, LinearSpec, LogSoftMaxSpec]) deriving (Generic, Show, Eq)
newtype MLP = MLP (HList [Linear, Relu, Linear, Relu, Linear, LogSoftMax]) deriving (Generic, Show, Eq)

mlpSpec :: MLPSpec
mlpSpec = MLPSpec $
  LinearSpec 784 64 .*.
  ReluSpec .*.
  LinearSpec 64 32 .*.
  ReluSpec .*.
  LinearSpec 32 10 .*.
  LogSoftMaxSpec .*.
  HNil

instance HasForward MLP Tensor Tensor where
  forward (MLP model) = forward model
  forwardStoch (MLP model) = forwardStoch model

instance Randomizable MLPSpec MLP where
  sample (MLPSpec spec) = MLP <$> sample spec

mlp :: MLP -> Tensor -> Tensor
mlp = forward
```

## Extract layer

## Test for each layer

For one input, take the outputs of all layers, then compare the shapes and values of all the layers.

## Overlay layer

## Concatenate layer

