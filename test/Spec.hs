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
{-# LANGUAGE DataKinds#-}
{-# LANGUAGE StandaloneDeriving#-}

module Main where

import Test.Hspec
import Torch.Compose
import Torch.Compose.NN
import Torch.Compose.Models
import Torch
import GHC.Generics hiding ((:+:))
import Data.HList
import Data.Proxy
import Data.Coerce

newtype MLPSpec = MLPSpec (HList [LinearSpec, ReluSpec, LinearSpec, ReluSpec, LinearSpec, LogSoftMaxSpec]) deriving (Generic, Show, Eq)
newtype MLP = MLP (HList [Linear, Relu, Linear, Relu, Linear, LogSoftMax]) deriving (Generic, Show, Eq)

deriving instance Eq Parameter
deriving instance Eq Linear

mlpSpec :: MLPSpec
mlpSpec = MLPSpec $
  LinearSpec 784 64 .*.
  ReluSpec .*.
  LinearSpec 64 32 .*.
  ReluSpec .*.
  LinearSpec 32 10 .*.
  LogSoftMaxSpec .*.
  HNil

brokenMlpSpec :: MLPSpec
brokenMlpSpec = MLPSpec $
  LinearSpec 784 64 .*.
  ReluSpec .*.
  LinearSpec 60 32 .*.
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

main :: IO ()
main = hspec $ do
  it "Creating a MLP in sequence" $ do
    model <- sample mlpSpec
    shape (mlp model (ones' [2,784])) `shouldBe` [2,10]
  it "Extract the first layer" $ do
    (MLP (model :: a)) <- sample mlpSpec
    let layer =  hHead model
    shape layer.weight.toDependent `shouldBe` [64,784]
  it "Extract the last layer" $ do
    (MLP (model :: a)) <- sample mlpSpec
    let layer =  hLast model
    layer `shouldBe` LogSoftMax
  it "Get, drop and add the last layer" $ do
    m <- sample mlpSpec
    let model' = dropLastLayer m
    let layer = getLastLayer model'
        lastLayer = getLastLayer m
    shape layer.weight.toDependent `shouldBe` [10,32]
    let model_rev = addLastLayer model' lastLayer
    coerce model_rev `shouldBe` m
  it "Extract all output shapes of MLP" $ do
    (MLP model) <- sample mlpSpec
    let input = ones' [2,784]
        output = forward model input
        outputs = toOutputs model input
        outputShapes = toOutputShapes model input
        exp = [2,64] .*. [2,64] .*. [2,32] .*. [2,32] .*. [2,10] .*. [2,10] .*. HNil
    outputShapes `shouldBe` exp
  it "Find a broken layer" $ do
    model <- sample brokenMlpSpec
    let input = ones' [2,784]
        output = forward model input
        outputs = toOutputs model input
        outputShapes = toMaybeOutputShapes model input
        exp = Just [2,64] .*. Just [2,64] .*. Nothing .*. Nothing .*. Nothing .*. Nothing .*. HNil
    outputShapes `shouldBe` exp
  it "Check vgg16Spec" $ do
    model <- sample (vgg16Spec 10)
    let input = ones' [2,3,128,128]
        outputShapes = toMaybeOutputShapes model input
        exp =
          Just [2,64,128,128] .*. Just [2,64,128,128] .*. Just [2,64,64,64] .*. Just [2,128,64,64] .*. Just [2,128,64,64] .*.
          Just [2,128,32,32] .*. Just [2,256,32,32] .*. Just [2,256,32,32] .*. Just [2,256,32,32] .*. Just [2,256,16,16] .*.
          Just [2,512,16,16] .*. Just [2,512,16,16] .*. Just [2,512,16,16] .*. Just [2,512,7,7] .*. Just [2,25088] .*.
          Just [2,4096] .*. Just [2,4096] .*. Just [2,4096] .*. Just [2,4096] .*. Just [2,4096] .*.
          Just [2,4096] .*. Just [2,10] .*. HNil
    outputShapes `shouldBe` exp
