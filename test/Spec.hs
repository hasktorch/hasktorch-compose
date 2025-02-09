{-# LANGUAGE AllowAmbiguousTypes#-}
{-# LANGUAGE DataKinds#-}
{-# LANGUAGE DeriveAnyClass#-}
{-# LANGUAGE DeriveGeneric#-}
{-# LANGUAGE DuplicateRecordFields#-}
{-# LANGUAGE FlexibleContexts#-}
{-# LANGUAGE FlexibleInstances#-}
{-# LANGUAGE FunctionalDependencies#-}
{-# LANGUAGE GADTs#-}
{-# LANGUAGE MultiParamTypeClasses#-}
{-# LANGUAGE OverloadedRecordDot#-}
{-# LANGUAGE RecordWildCards#-}
{-# LANGUAGE ScopedTypeVariables#-}
{-# LANGUAGE StandaloneDeriving#-}
{-# LANGUAGE TypeApplications#-}
{-# LANGUAGE TypeFamilies#-}
{-# LANGUAGE TypeOperators#-}
{-# LANGUAGE UndecidableInstances#-}

module Main where

import Test.Hspec
import Torch.Compose
import Torch.Compose.NN
import Torch.Compose.Models
import Torch
import Torch.Lens
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
  it "Check the shape of vgg16Spec" $ do
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
  it "Add zero to fine-tune the last layer only with mergeParameters" $ do
    m0 <- sample mlpSpec
    m1' <- sample mlpSpec
    let layer0 = getLastLayer m0
        zero1 = over (types @Tensor) (ones' . shape) $ getLastLayer m1'
        model' = addLastLayer (dropLastLayer m0) (mergeParameters (+) layer0 zero1)
    return ()
  it "Fanin" $ do
    m0 <- sample mlpSpec
    m1 <- sample mlpSpec
    let l0 = getFirstLayer m0
        l1 = getFirstLayer m1
        fin = Fanin l0 l1
        model' = HCons fin $ dropFirstLayer m0
        input = ones' [2,784]
        out = forward model' (input,input)
    shape out `shouldBe` [2,10]
  it "Fanout" $ do
    m0 <- sample (LinearSpec 10 2)
    m1 <- sample (LinearSpec 10 3)
    let fout = Fanout m0 m1
        input = ones' [1,10]
        (out0,out1) = forward fout input
    shape out0 `shouldBe` [1,2]
    shape out1 `shouldBe` [1,3]
  it "Split layers" $ do
    m0 <- sample mlpSpec
    let (h, t) = splitLayers @2 m0
        input0 = ones'  [1,784]
        input1 = ones'  [1,64]
        output0 = forward h input0
        output1 = forward t input1
    shape output0 `shouldBe` [1,64]
    shape output1 `shouldBe` [1,10]
