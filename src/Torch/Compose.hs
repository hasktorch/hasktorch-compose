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

module Torch.Compose where

import Torch
import Torch.NN
import Torch.Functional
import GHC.Generics hiding ((:+:))
import System.IO.Unsafe (unsafePerformIO)

data (:>>:) a b = Forward
  { head :: a
  , tail :: b
  } deriving (Show, Generic)

data (://:) a b = Fanout
  { head :: a
  , tail :: b
  } deriving (Show, Generic)

data (:+:) a b = Fanin
  { head :: a
  , tail :: b
  } deriving (Show, Generic)

data (:++:) a b = Concat
  { head :: a
  , tail :: b
  } deriving (Show, Generic)

data ShortcutSpec a = ShortcutSpec a deriving (Show, Generic)

data Shortcut a = Shortcut a deriving (Show, Generic, Parameterized)

instance (Randomizable a b) => Randomizable (ShortcutSpec a) (Shortcut b) where
  sample (ShortcutSpec s) = Shortcut <$> sample s

data ReplicateSpec b = ReplicateSpec Int b deriving (Show, Generic)
data Replicate b = Replicate [b] deriving (Show, Generic)

instance (Randomizable b c) => Randomizable (ReplicateSpec b) (Replicate c) where
  sample (ReplicateSpec n s) = Replicate <$> sequence (replicate n (sample s))

instance (HasForward a b b) => HasForward (Replicate a) b b where
  forward (Replicate []) input = input
  forward (Replicate (a:ax)) input = forward (Replicate ax) (forward a input)
  forwardStoch (Replicate []) input = pure input
  forwardStoch (Replicate (a:ax)) input = forwardStoch (Replicate ax) =<< forwardStoch a input

instance (HasForward f a b, HasForward g b c) => HasForward (f :>>: g) a c where
  forward (Forward f g) a = forward g (forward f a)
  forwardStoch (Forward f g) a = forwardStoch f a >>= forwardStoch g 

instance (HasForward f a b, HasForward g a c) => HasForward (f ://: g) a (b,c) where
  forward (Fanout f g) a = (forward f a, forward g a)
  forwardStoch (Fanout f g) a = (,) <$> forwardStoch f a <*> forwardStoch g a

instance (Num c, HasForward f a c, HasForward g b c) => HasForward (f :+: g) (a,b) c where
  forward (Fanin f g) (a,b) = forward f a + forward g b
  forwardStoch (Fanin f g) (a,b) = do
    c <- forwardStoch f a
    c' <- forwardStoch g b
    return (c + c')

instance (Randomizable spec0 f0, Randomizable spec1 f1) => Randomizable (spec0 :>>: spec1) (f0 :>>: f1) where
  sample (Forward s0 s1) = do
    f0 <- sample s0
    f1 <- sample s1
    return (Forward f0 f1)

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


instance (HasForward a Tensor Tensor) => HasForward (Shortcut a) Tensor Tensor where
  forward (Shortcut f) input = forward f input + input
  forwardStoch (Shortcut f) input = do
    f' <- forwardStoch f input
    return $ f' + input

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

vgg16Spec numClass =
  let maxPool2dSpec = MaxPool2dSpec
        { kernelSize = (3,3)
        , stride = (2,2)
        , padding = (1,1)
        , dilation = (0,0)
        , ceilMode = Ceil
        }
      vggClassifierSpec =
        Forward (LinearSpec (512 * 7 * 7) 4096) $
        Forward ReluSpec $
        Forward DropoutSpec $
        Forward (LinearSpec 4096 4096) $
        Forward ReluSpec $
        Forward DropoutSpec $
        Forward (LinearSpec 4096 numClass)
  in
  Forward (Conv2dSpec 3 64 3 3) $
  Forward (Conv2dSpec 64 64 3 3) $
  Forward maxPool2dSpec $ 
  Forward (Conv2dSpec 64 128 3 3) $
  Forward (Conv2dSpec 128 128 3 3) $
  Forward maxPool2dSpec $ 
  Forward (Conv2dSpec 128 256 3 3) $
  Forward (Conv2dSpec 256 256 3 3) $
  Forward (Conv2dSpec 256 256 3 3) $
  Forward maxPool2dSpec $ 
  Forward (Conv2dSpec 256 512 3 3) $
  Forward (Conv2dSpec 512 512 3 3) $
  Forward (Conv2dSpec 512 512 3 3) $
  Forward (AdaptiveAvgPool2dSpec (7,7)) $ 
  Forward (ReshapeSpec [1,512*7*7]) $
  Forward vggClassifierSpec

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
  
