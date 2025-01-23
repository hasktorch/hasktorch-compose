{-# LANGUAGE TypeOperators#-}
{-# LANGUAGE FlexibleInstances#-}
{-# LANGUAGE MultiParamTypeClasses#-}
{-# LANGUAGE UndecidableInstances#-}
{-# LANGUAGE DeriveGeneric#-}
{-# LANGUAGE DeriveAnyClass#-}
{-# LANGUAGE DuplicateRecordFields#-}
{-# LANGUAGE RecordWildCards#-}
{-# LANGUAGE ScopedTypeVariables#-}
{-# LANGUAGE TypeApplications#-}
{-# LANGUAGE GADTs#-}
{-# LANGUAGE OverloadedRecordDot#-}
{-# LANGUAGE FlexibleContexts#-}
{-# LANGUAGE TypeFamilies#-}
{-# LANGUAGE AllowAmbiguousTypes#-}
{-# LANGUAGE FunctionalDependencies#-}


module Main where

import Test.Hspec
import Torch.Compose
import Torch.Compose.NN
import Torch
import GHC.Generics hiding ((:+:))

newtype MLPSpec = MLPSpec (LinearSpec :>>: (ReluSpec :>>: (LinearSpec :>>: (ReluSpec :>>: LinearSpec)))) deriving (Generic)
newtype MLP = MLP (Linear :>>: (Relu :>>: (Linear :>>: (Relu :>>: Linear)))) deriving (Generic)

mlpSpec :: MLPSpec
mlpSpec = MLPSpec $
  Forward (LinearSpec 784 64) $
  Forward ReluSpec $
  Forward (LinearSpec 64 32) $
  Forward ReluSpec $
  LinearSpec 32 10

instance HasForward MLP Tensor Tensor where
  forward (MLP model) = forward model
  forwardStoch (MLP model) = forwardStoch model

instance Randomizable MLPSpec MLP where
  sample (MLPSpec spec) = MLP <$> sample spec

mlp :: MLP -> Tensor -> Tensor
mlp model input =
  logSoftmax (Dim 1) $ forward model input

main :: IO ()
main = hspec $ do
  it "Creating a MLP in sequence" $ do
    model <- sample mlpSpec
    shape (mlp model (ones' [2,784])) `shouldBe` [2,10]
  it "Extract the first layer" $ do
    (MLP (model :: a)) <- sample mlpSpec
    let layer =  getFirst model :: FirstLayer a
    shape layer.weight.toDependent `shouldBe` [64,784]
  it "Extract the last layer" $ do
    (MLP (model :: a)) <- sample mlpSpec
    let layer =  getLast model :: LastLayer a
    shape layer.weight.toDependent `shouldBe` [10,32]
  it "Extract all output shapes" $ do
    (MLP (model :: a)) <- sample mlpSpec
    let out = toOutputShapes model (ones' [2,784])
        exp = Forward [2,64] $ Forward [2,64] $ Forward [2,32] $ Forward [2,32] [2,10]
    out `shouldBe` exp
    
