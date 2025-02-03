# Hasktorch Compose

In Hasktorch, model specifications, values, and inference are defined separately. This often necessitates combining commonly used models. For example, three linear layers may need to be connected in sequence, or a new linear layer could be added to an existing model. Hasktorch Compose provides a straightforward way to compose such models.

In addition to simple model composition, this library aims to support extracting parts of models and sharing parameters between different models, such as ControlNet and RoLa. Both an untyped API and a typed API are planned, with initial development focused on the untyped API.

Hasktorch Compose is an experimental library built on top of [hasktorch-skeleton](https://github.com/hasktorch/hasktorch-skeleton).

**Features:**
- [x] Sequential composition of layers (using HList)
- [x] Extracting individual layers 
- [x] Adding/Dropping/Replacing layers in a sequence
- [x] Inspecting layer output shapes
- [x] Concatenate layer (`:++:`)
- [x] Fanout layer (`://:`)
- [x] Fanin layer (`:+:`)
- [x] Shortcut layer
- [x] Replicate layer
- [ ] Test for each layer
- [ ] Overlay layer


# Examples

## Sequential Composition

Use the `.*.` operator of HList to join layer specifications. 
This example shows how to create a simple Multi-Layer Perceptron (MLP) by combining `LinearSpec` layers with `ReluSpec` activations.

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

## Extracting Layers

You can extract the first or last layer of a composed model using `getFirstLayer` and `getLastLayer` respectively. You can also drop the first or last layer with `dropFirstLayer` and `dropLastLayer` functions.

```haskell
-- Assume 'model' is an instance of `MLP`
(MLP (model :: a)) <- sample mlpSpec
let firstLayer = getFirstLayer model -- Get the first Linear layer
let lastLayer = getLastLayer model  -- Get the LogSoftmax layer
let modelWithoutLast = dropLastLayer model
```

## Modifying Layers

Layers can be added to a sequence using `addLastLayer`.

```haskell
-- Assume 'model' is an instance of `MLP`
model <- sample mlpSpec
let modelWithoutLast = dropLastLayer model
let lastLayer = getLastLayer model
let modifiedModel = addLastLayer modelWithoutLast lastLayer -- add the last layer back
```

## Inspecting Output Shapes

The `toOutputShapes` function allows you to get the shapes of each layer's output for a given input. This is useful for debugging and understanding the data flow in a model.

```haskell
-- Assume 'model' is an instance of `MLP`
(MLP model) <- sample mlpSpec
let input = ones' [2,784]
let outputShapes = toOutputShapes model input
-- outputShapes will be a HList containing the shape of each layer's output.
```

## Concatenate Layer

The `Concat` type (using `:++:` as the infix constructor) allows you to combine two models that operate on different inputs.

```haskell
-- Assume 'm0' and 'm1' are models, and 'a0' and 'a1' are inputs.
let concatenatedModel = Concat m0 m1
let (b0, b1) = forward concatenatedModel (a0, a1)
```

## Fanout Layer

The `Fanout` type (using `://:` as the infix constructor)  allows you to apply different models to the same input.

```haskell
-- Assume 'm0' and 'm1' are models, and 'a' is the input.
let fanoutModel = Fanout m0 m1
let (b0, b1) = forward fanoutModel a
```

## Fanin Layer

The `Fanin` type (using `:+:` as the infix constructor)  allows you to combine the outputs of different models using element-wise addition.

```haskell
-- Assume 'm0' and 'm1' are models, and 'a' and 'b' are inputs.
let faninModel = Fanin m0 m1
let c = forward faninModel (a, b)
```

## Shortcut Layer
The `Shortcut` layer allows you to implement a residual connection by adding the input to the output of a given model.

```haskell
-- Assume 'model' is an instance of some model
let shortcutModel = Shortcut model
let output = forward shortcutModel input
-- output == forward model input + input
```

## Replicate Layer
The `Replicate` layer replicates a given model `n` times and applies each one sequentially.

```haskell
-- Assume 'model' is an instance of some model
let replicatedModel = Replicate n model
let output = forward replicatedModel input
-- output == forward model (forward model (... (forward model input) ...))
-- where 'forward model' is applied n times
```

## Merging Parameters

The `mergeParameters` function can be used to combine the parameters of two models. In the example below, only the last layer of the second model is added to the last layer of the first model.

```haskell
m0 <- sample mlpSpec
m1' <- sample mlpSpec
let layer0 = getLastLayer m0
let zero1 = over (types @Tensor) (ones' . shape) $ getLastLayer m1'
let model' = addLastLayer (dropLastLayer m0) (mergeParameters (+) layer0 zero1)
```

