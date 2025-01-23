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
{-# LANGUAGE NoFieldSelectors#-}
{-# LANGUAGE TypeFamilies#-}

module Torch.Compose.NN where

import Torch
import Torch.Compose
import System.IO.Unsafe (unsafePerformIO)
import GHC.Generics hiding ((:+:))

data ReluSpec = ReluSpec deriving (Generic, Show, Eq)
data Relu = Relu deriving (Generic, Parameterized, Show, Eq)
instance Randomizable ReluSpec Relu where
  sample _ = pure Relu

instance HasForward Relu Tensor Tensor where
  forward _ = relu
  forwardStoch _ i = pure $ relu i


data DropoutSpec where
  DropoutSpec ::
    {dropoutProbSpec :: Double} ->
    DropoutSpec
  deriving (Show, Eq)

data Dropout = Dropout
  { dropoutProb :: Double
  }
  deriving (Show, Generic, Parameterized)

instance HasForward Dropout Tensor Tensor where
  forward Dropout{..} input = unsafePerformIO $ dropout dropoutProb False input
  forwardStoch Dropout{..} = dropout dropoutProb True

data MaxPool2dSpec = MaxPool2dSpec
  { kernelSize :: (Int,Int)
  , stride :: (Int,Int)
  , padding :: (Int,Int)
  , dilation :: (Int,Int)
  , ceilMode :: CeilMode
  }
  deriving (Show, Eq)

data MaxPool2d = MaxPool2d
  { spec :: MaxPool2dSpec
  }
  
instance HasForward MaxPool2d Tensor Tensor where
  forward param =
    let p = param.spec
    in maxPool2d p.kernelSize p.stride p.padding p.dilation p.ceilMode
  forwardStoch = (pure .) . forward

data AdaptiveAvgPool2dSpec = AdaptiveAvgPool2dSpec
  { outputSize :: (Int,Int)
  }
  deriving (Show, Eq)

data AdaptiveAvgPool2d = AdaptiveAvgPool2d
  { spec :: AdaptiveAvgPool2dSpec
  }

instance HasForward AdaptiveAvgPool2d Tensor Tensor where
  forward param = adaptiveAvgPool2d (param.spec.outputSize)
  forwardStoch = (pure .) . forward

data ReshapeSpec = ReshapeSpec
  { shape :: [Int]
  }

data Reshape = Reshape
  { spec :: ReshapeSpec
  }

instance HasForward Reshape Tensor Tensor where
  forward param input = reshape param.spec.shape input
  forwardStoch = (pure .) . forward


data BatchNorm2dSpec
  = BatchNorm2dSpec
  { channelSize :: Int
  }
  deriving (Show, Generic)

data
  BatchNorm2d
  = BatchNorm2d
  { spec :: BatchNorm2dSpec
  , weight :: Parameter
  , bias :: Parameter
  , runningMean :: MutableTensor
  , runningVar :: MutableTensor
  }
  deriving (Show, Generic)

instance Randomizable BatchNorm2dSpec BatchNorm2d where
  sample spec'@BatchNorm2dSpec{..} = do
    spec <- pure spec'
    weight <- makeIndependent =<< randnIO' [channelSize]
    bias <- makeIndependent =<< randnIO' [channelSize]
    runningMean <- newMutableTensor $ zeros' [channelSize]
    runningVar <- newMutableTensor $ ones' [channelSize]
    pure BatchNorm2d{..}

resnetSpec numClass =
  Forward (Conv2dSpec 3 64 7 7) $
  Forward (BatchNorm2dSpec 64) $
  Forward ReluSpec $
  Forward (Conv2dSpec 64 64 3 3)
  

instance HasOutputs Linear Tensor where
  type Outputs Linear Tensor = Tensor
  toOutputs = forward

instance HasOutputShapes Linear Tensor where
  type OutputShapes Linear Tensor = [Int]
  toOutputShapes model a = shape $ forward model a

instance HasInputs Linear Tensor where
  type Inputs Linear Tensor = Tensor
  toInputs _ a = a

instance HasOutputs Relu Tensor where
  type Outputs Relu Tensor = Tensor
  toOutputs = forward

instance HasInputs Relu Tensor where
  type Inputs Relu Tensor = Tensor
  toInputs _ a = a

instance HasOutputShapes Relu Tensor where
  type OutputShapes Relu Tensor = [Int]
  toOutputShapes model a = shape $ forward model a

instance HasForwardAssoc Linear Tensor where
  type ForwardResult Linear Tensor = Tensor
  forwardAssoc = forward

instance HasForwardAssoc Relu Tensor where
  type ForwardResult Relu Tensor = Tensor
  forwardAssoc = forward

