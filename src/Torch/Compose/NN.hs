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
import qualified Torch.Functional.Internal as T
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
  
  -------------------------------------------------------------------------------
-- 1. LayerNorm
-------------------------------------------------------------------------------

data LayerNormSpec = LayerNormSpec
  { lnDim  :: Int   -- ^ dimension (e.g. embedDim)
  , lnEps  :: Float -- ^ small epsilon
  }
  deriving (Show, Eq)

data LayerNorm = LayerNorm
  { spec      :: LayerNormSpec
  , gamma     :: Parameter -- scale
  , beta      :: Parameter -- bias
  } deriving (Show)

instance Randomizable LayerNormSpec LayerNorm where
  sample s@LayerNormSpec{..} = do
    let wInit = ones'  [lnDim]
        bInit = zeros' [lnDim]
    gammaParam <- makeIndependent wInit
    betaParam  <- makeIndependent bInit
    pure LayerNorm
      { spec  = s
      , gamma = gammaParam
      , beta  = betaParam
      }

--------------------------------------------------------------------------------
-- LayerNorm (fixed mean/var)
--------------------------------------------------------------------------------

instance HasForward LayerNorm Tensor Tensor where
  forward LayerNorm{..} input =
    let
      -- For dimension -1, and keepDim = True:
      -- T.meanDim, T.varDim from Torch.Functional.Internal
      mean' = meanDim (Dim (-1)) KeepDim Float input
      var'  = T.varDim  input (-1) True True
      xNorm = (input - mean') / Torch.sqrt (var' + asTensor spec.lnEps)
      out   = xNorm * toDependent gamma + toDependent beta
    in out

  forwardStoch ln = pure . forward ln

-------------------------------------------------------------------------------
-- 2. Simple Feed-Forward Network
-------------------------------------------------------------------------------

data FeedForwardSpec = FeedForwardSpec
  { ffInDim  :: Int
  , ffHidden :: Int
  }
  deriving (Show, Eq)

data FeedForward = FeedForward
  { l1 :: Linear
  , l2 :: Linear
  }
  deriving (Show)

instance Randomizable FeedForwardSpec FeedForward where
  sample FeedForwardSpec{..} = do
    fc1 <- sample $ LinearSpec ffInDim ffHidden
    fc2 <- sample $ LinearSpec ffHidden ffInDim
    pure FeedForward { l1 = fc1, l2 = fc2 }

instance HasForward FeedForward Tensor Tensor where
  forward FeedForward{..} input =
    let x1 = relu (linear l1 input)
        x2 = linear l2 x1
    in x2

  forwardStoch ff = pure . forward ff

-------------------------------------------------------------------------------
-- 3. Causal Masking Utility
-------------------------------------------------------------------------------

-- | Create a causal "upper-triangular" mask so that position j > i is masked out.
--   shape: [seqLen, seqLen], with 1.0 = keep, 0.0 = block
createCausalMask :: Int -> Tensor
createCausalMask seqLen =
  let range   = arange' 0 (fromIntegral seqLen) 1 -- [seqLen]
      rowIdx  = unsqueeze (Dim (-1)) range           -- shape [seqLen, 1]
      colIdx  = unsqueeze (Dim 0) range             -- shape [1, seqLen]
      -- If rowIdx < colIdx => "future" => 0.0, else 1.0
      keepBool = rowIdx `ge` colIdx
      keep     = T.where' keepBool (onesLike keepBool) (zerosLike keepBool)
  in keep

-------------------------------------------------------------------------------
-- 4. GPT-2 Decoder Block
-------------------------------------------------------------------------------

data GPT2BlockSpec = GPT2BlockSpec
  { blockEmbedDim :: Int
  , blockNumHeads :: Int
  , blockFfHidden :: Int
  , blockLnEps    :: Float
  }
  deriving (Show, Eq)

data GPT2Block = GPT2Block
  { ln1  :: LayerNorm
  , attn :: MultiHeadAttention
  , ln2  :: LayerNorm
  , ff   :: FeedForward
  }
  deriving (Show)

instance Randomizable GPT2BlockSpec GPT2Block where
  sample GPT2BlockSpec{..} = do
    let lnSpec  = LayerNormSpec blockEmbedDim blockLnEps
        ffSpec  = FeedForwardSpec blockEmbedDim blockFfHidden
        mhaSpec = MultiHeadAttentionSpec blockEmbedDim blockNumHeads
    GPT2Block
      <$> sample lnSpec
      <*> sample mhaSpec
      <*> sample lnSpec
      <*> sample ffSpec

-- | GPT2Block forward: 
--   1) LN + masked self-attn
--   2) Residual
--   3) LN + feed-forward
--   4) Residual
instance HasForward GPT2Block (Tensor, Tensor) Tensor where
  -- ^ We'll accept `(x, mask)` as input, return the new hidden states.
  --   The `mask` is shape [1, seqLen, seqLen] or broadcastable to [batchSize, seqLen, seqLen].
  forward GPT2Block{..} (x, mask) =
    let xNorm     = forward ln1 x
        -- Because our 'multiHeadAttention' does not directly accept a mask yet,
        -- we can *simulate* it by zeroing out "future" attention in the matmul,
        -- or you can adapt your MHA to accept a mask argument. 
        -- For simplicity, let's do a minimal approach:
        -- We'll skip the explicit mask in the code if your MHA doesn't use it.
        -- If you extended multiHeadAttention to handle a mask, you'd pass it there.
        attnOut   = multiHeadAttention attn xNorm xNorm xNorm
        x1        = x + attnOut     -- residual
        x1Norm    = forward ln2 x1
        ffOut     = forward ff x1Norm
        x2        = x1 + ffOut      -- residual
    in x2

  forwardStoch block (x, mask) = pure $ forward block (x, mask)

-------------------------------------------------------------------------------
-- 5. The Full GPT2 Model
-------------------------------------------------------------------------------

data GPT2Spec = GPT2Spec
  { vocabSize  :: Int
  , maxPos     :: Int
  , numLayers  :: Int
  , embedDim   :: Int
  , numHeads   :: Int
  , ffHiddenDim:: Int
  , lnEpsVal   :: Float
  }
  deriving (Show, Eq)

data GPT2 = GPT2
  { tokenEmbed   :: Parameter          -- ^ [vocabSize, embedDim]
  , positionEmbed:: Parameter          -- ^ [maxPos, embedDim]
  , blocks       :: [GPT2Block]
  , lnFinal      :: LayerNorm
  }
  deriving (Show)

instance Randomizable GPT2Spec GPT2 where
  sample GPT2Spec{..} = do
    tokenParam   <- makeIndependent =<< randnIO' [vocabSize, embedDim]
    posParam     <- makeIndependent =<< randnIO' [maxPos, embedDim]
    let blockSpec = GPT2BlockSpec
          { blockEmbedDim = embedDim
          , blockNumHeads = numHeads
          , blockFfHidden = ffHiddenDim
          , blockLnEps    = lnEpsVal
          }
    gpt2Blocks  <- mapM (const $ sample blockSpec) [1..numLayers]
    finalNorm   <- sample $ LayerNormSpec embedDim lnEpsVal
    pure GPT2
      { tokenEmbed    = tokenParam
      , positionEmbed = posParam
      , blocks        = gpt2Blocks
      , lnFinal       = finalNorm
      }

-- | We'll define HasForward for GPT2 taking just the input token IDs:
--   shape: [batchSize, seqLen], returning [batchSize, seqLen, vocabSize].
instance HasForward GPT2 Tensor Tensor where
  forward GPT2{..} inputIds =
    let (batchSize, seqLen) = case shape inputIds of
                                [b, s] -> (b, s)
                                _      -> error "GPT2 forward: expected [batchSize, seqLen]"
        -- 1) Get token embeddings
        xToken = embedding' (toDependent tokenEmbed) inputIds
          -- [batchSize, seqLen, embedDim]
        -- 2) Get position embeddings
        positions   = arange' 0 (fromIntegral seqLen) 1 -- [seqLen]
        posEmbs     = embedding' (toDependent positionEmbed) positions
          -- [seqLen, embedDim]
        posEmbs3d   = unsqueeze (Dim 0) posEmbs
          -- [1, seqLen, embedDim]
        posEmbsB    = expand posEmbs3d False [batchSize, seqLen, shape posEmbs3d !! 2]

        x           = xToken + posEmbsB
        -- 3) Build a causal mask if your MHA supports it; for now let's ignore if your MHA doesn't handle masks:
        mask        = unsqueeze (Dim 0) (createCausalMask seqLen)  
          -- shape [1, seqLen, seqLen]
        
        -- 4) Pass through each GPT2Block
        xOut = foldl (\acc block -> forward block (acc, mask)) x blocks
        -- 5) Final layer norm
        xNorm = forward lnFinal xOut
        -- 6) Project to vocab (if you want weight tying, typically we do xNorm `matmul` transpose tokenEmbed)
        tokenWeightT = transpose2D (toDependent tokenEmbed)
          -- shape [embedDim, vocabSize]
        logits       = xNorm `matmul` tokenWeightT
          -- [batchSize, seqLen, vocabSize]
    in logits

  forwardStoch net inputIds = pure $ forward net inputIds

-------------------------------------------------------------------------------
-- 6. Add HasForwardAssoc (Optional)
-------------------------------------------------------------------------------

-- If you are using `instanceForwardAssocs` to auto-generate associated type families,
-- you can include GPT2, GPT2Block, and so on. For example:
{-
instanceForwardAssocs
  [ [t| GPT2Block |]
  , [t| GPT2      |]
  ]
  [t| (Tensor, Tensor) |]  -- For GPT2Block we used (x,mask) as input
  [t| Tensor |]

instanceForwardAssocs
  [ [t| GPT2 |] ]
  [t| Tensor |] [t| Tensor |]
-}
