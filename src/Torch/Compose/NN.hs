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
{-# LANGUAGE DataKinds#-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE QuasiQuotes #-}

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

newtype DropoutSpec = DropoutSpec
  { dropoutProb :: Double
  } deriving (Show, Eq)

newtype Dropout = Dropout
  { spec :: DropoutSpec
  }
  deriving (Show, Generic)

instance HasForward Dropout Tensor Tensor where
  forward Dropout{..} input = unsafePerformIO $ dropout spec.dropoutProb False input
  forwardStoch Dropout{..} = dropout spec.dropoutProb True

instance Randomizable DropoutSpec Dropout where
  sample = pure . Dropout

data MaxPool2dSpec = MaxPool2dSpec
  { kernelSize :: (Int,Int)
  , stride :: (Int,Int)
  , padding :: (Int,Int)
  , dilation :: (Int,Int)
  , ceilMode :: CeilMode
  }
  deriving (Show, Eq)

newtype MaxPool2d = MaxPool2d
  { spec :: MaxPool2dSpec
  }
  
instance HasForward MaxPool2d Tensor Tensor where
  forward param =
    let p = param.spec
    in maxPool2d p.kernelSize p.stride p.padding p.dilation p.ceilMode
  forwardStoch = (pure .) . forward

instance Randomizable MaxPool2dSpec MaxPool2d where
  sample = pure . MaxPool2d

newtype AdaptiveAvgPool2dSpec = AdaptiveAvgPool2dSpec
  { outputSize :: (Int,Int)
  } deriving (Show, Eq)

newtype AdaptiveAvgPool2d = AdaptiveAvgPool2d
  { spec :: AdaptiveAvgPool2dSpec
  }

instance HasForward AdaptiveAvgPool2d Tensor Tensor where
  forward param = adaptiveAvgPool2d param.spec.outputSize
  forwardStoch = (pure .) . forward

instance Randomizable AdaptiveAvgPool2dSpec AdaptiveAvgPool2d where
  sample = pure . AdaptiveAvgPool2d

newtype ReshapeSpec = ReshapeSpec
  { shape :: [Int]
  }

newtype Reshape = Reshape
  { spec :: ReshapeSpec
  }

instance HasForward Reshape Tensor Tensor where
  forward param = reshape param.spec.shape
  forwardStoch = (pure .) . forward

instance Randomizable ReshapeSpec Reshape where
  sample = pure . Reshape

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
    let spec = spec'
    weight <- makeIndependent =<< randnIO' [channelSize]
    bias <- makeIndependent =<< randnIO' [channelSize]
    runningMean <- newMutableTensor $ zeros' [channelSize]
    runningVar <- newMutableTensor $ ones' [channelSize]
    pure BatchNorm2d{..}

data LogSoftMaxSpec = LogSoftMaxSpec deriving (Generic, Show, Eq)
data LogSoftMax = LogSoftMax deriving (Generic, Parameterized, Show, Eq)
instance Randomizable LogSoftMaxSpec LogSoftMax where
  sample _ = pure LogSoftMax

instance HasForward LogSoftMax Tensor Tensor where
  forward _ = logSoftmax (Dim 1)
  forwardStoch _ i = pure $ logSoftmax (Dim 1) i

data Conv2dSpec' = Conv2dSpec'
  { inputChannelSize2d :: Int
  , outputChannelSize2d :: Int
  , kernelHeight2d :: Int
  , kernelWidth2d :: Int
  , stride :: (Int, Int)
  , padding ::(Int, Int)
  } deriving (Generic, Show, Eq)

data Conv2d' = Conv2d'
  { spec :: Conv2dSpec'
  , params :: Conv2d
  } deriving (Generic, Show)

instance Parameterized Conv2d' where
  flattenParameters d = flattenParameters d.params
  _replaceParameters d = (\p -> d {params = p}) <$> _replaceParameters d.params

instance HasForward Conv2d' Tensor Tensor where
  forward params = conv2dForward params.params params.spec.stride params.spec.padding
  forwardStoch params input = pure $ forward params input

instance Randomizable Conv2dSpec' Conv2d' where
  sample spec = do
    a <- sample $ Conv2dSpec
         { inputChannelSize2d = spec.inputChannelSize2d
         , outputChannelSize2d = spec.outputChannelSize2d
         , kernelHeight2d = spec.kernelHeight2d
         , kernelWidth2d = spec.kernelWidth2d
         }
    return $ Conv2d'
      { spec = spec
      , params = a
      }


--------------------------------------------------------------------------------
-- Multi-Head Attention Data Structures
--------------------------------------------------------------------------------

-- | Specification for initializing a MultiHeadAttention module.
data MultiHeadAttentionSpec = MultiHeadAttentionSpec
  { mhaEmbedDim :: Int     -- ^ Model embedding dimension
  , mhaNumHeads :: Int     -- ^ Number of attention heads
  } deriving (Show, Eq)

-- | Data type that holds parameters for Multi-Head Attention.
data MultiHeadAttention = MultiHeadAttention
  { wQ      :: Linear      -- ^ Linear projection for the queries
  , wK      :: Linear      -- ^ Linear projection for the keys
  , wV      :: Linear      -- ^ Linear projection for the values
  , wO      :: Linear      -- ^ Final linear projection after combining heads
  , headDim :: Int         -- ^ Dimension per head = embedDim / numHeads
  , nHeads  :: Int         -- ^ Number of attention heads
  } deriving (Show)

-- | Create random parameters for Multi-Head Attention given the specification.
instance Randomizable MultiHeadAttentionSpec MultiHeadAttention where
  sample MultiHeadAttentionSpec{..} = do
    let headDim = mhaEmbedDim `Prelude.div` mhaNumHeads
    wQ' <- sample $ LinearSpec mhaEmbedDim mhaEmbedDim
    wK' <- sample $ LinearSpec mhaEmbedDim mhaEmbedDim
    wV' <- sample $ LinearSpec mhaEmbedDim mhaEmbedDim
    wO' <- sample $ LinearSpec mhaEmbedDim mhaEmbedDim
    return $ MultiHeadAttention
      { wQ      = wQ'
      , wK      = wK'
      , wV      = wV'
      , wO      = wO'
      , headDim = headDim
      , nHeads  = mhaNumHeads
      }

--------------------------------------------------------------------------------
-- Forward Pass (Scaled Dot-Product Attention + Multi-Head Logic)
--------------------------------------------------------------------------------

-- | Compute scaled dot-product attention for query, key, value tensors.
--   The typical shape for q, k, v is:
--      [batchSize, numHeads, seqLen, headDim]
--
--   Returns: [batchSize, numHeads, seqLen, headDim]
scaledDotProductAttention
  :: Tensor  -- ^ Queries (q)
  -> Tensor  -- ^ Keys (k)
  -> Tensor  -- ^ Values (v)
  -> Tensor  -- ^ Output (contextual embeddings)
scaledDotProductAttention q k v =
  let -- q*k^T -> [batchSize, numHeads, seqLen, seqLen]
      dk          = fromIntegral (shape q !! 3) -- headDim
      scores      = (q `matmul` transpose2D k) / Torch.sqrt (asTensor (dk :: Float))
      attnWeights = softmax (Dim (-1)) scores         -- softmax over last dim (seqLen)
      output      = attnWeights `matmul` v      -- multiply by values
  in output

-- | Forward pass for Multi-Head Attention (without any mask or dropout, minimal).
multiHeadAttention
  :: MultiHeadAttention
  -> Tensor  -- ^ Input queries [batchSize, seqLen, embedDim]
  -> Tensor  -- ^ Input keys    [batchSize, seqLen, embedDim]
  -> Tensor  -- ^ Input values  [batchSize, seqLen, embedDim]
  -> Tensor  -- ^ Output        [batchSize, seqLen, embedDim]
multiHeadAttention MultiHeadAttention{..} queries keys values =
  let
    -- Project inputs to Q, K, V space
    q = linear wQ queries   -- [batchSize, seqLen, embedDim]
    k = linear wK keys      -- [batchSize, seqLen, embedDim]
    v = linear wV values    -- [batchSize, seqLen, embedDim]

    batchSize = shape queries !! 0
    seqLen    = shape queries !! 1

    -- Reshape for multi-head: [batchSize, seqLen, nHeads*headDim]
    -- -> [batchSize, seqLen, nHeads, headDim]
    -- -> [batchSize, nHeads, seqLen, headDim]
    reshapeForHeads t =
      let t' = view [batchSize, seqLen, nHeads*headDim] t
          t''= view [batchSize, seqLen, nHeads, headDim] t'
      in permute [0,2,1,3] t''  -- reorder dimensions to [batchSize, nHeads, seqLen, headDim]

    qHeads = reshapeForHeads q
    kHeads = reshapeForHeads k
    vHeads = reshapeForHeads v

    -- Apply scaled dot-product attention
    attnOutput = scaledDotProductAttention qHeads kHeads vHeads
        -- shape: [batchSize, nHeads, seqLen, headDim]

    -- Convert back: [batchSize, nHeads, seqLen, headDim]
    -- -> [batchSize, seqLen, nHeads, headDim]
    -- -> [batchSize, seqLen, nHeads*headDim]
    attnOutputTrans = permute [0,2,1,3] attnOutput
    combinedHeads   = view [batchSize, seqLen, nHeads*headDim] attnOutputTrans

    -- Final linear
    out = linear wO combinedHeads  -- [batchSize, seqLen, embedDim]
  in out



-- Generate HasForwardAssoc instances from HasForward instances.
instanceForwardAssocs
  [ [t| Linear |]
  , [t| Relu |]
  , [t| LogSoftMax |]
  , [t| AdaptiveAvgPool2d |]
  , [t| MaxPool2d |]
  , [t| Reshape |]
  , [t| Dropout |]
  , [t| Conv2d' |]
  ]
  [t| Tensor |] [t| Tensor |]
  
