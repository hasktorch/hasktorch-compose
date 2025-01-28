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
{-# LANGUAGE DataKinds#-}

module Torch.Compose.Models where

import Torch
import Torch.Compose
import Torch.Compose.NN
import GHC.Generics hiding ((:+:))
import Data.HList

vgg16Spec numClass =
  let maxPool2dSpec = MaxPool2dSpec
        { kernelSize = (3,3)
        , stride = (2,2)
        , padding = (1,1)
        , dilation = (0,0)
        , ceilMode = Ceil
        }
      vggClassifierSpec =
        LinearSpec (512 * 7 * 7) 4096 .*.
        ReluSpec .*.
        DropoutSpec 0.5 .*.
        LinearSpec 4096 4096 .*.
        ReluSpec .*.
        DropoutSpec 0.5 .*.
        LinearSpec 4096 numClass .*.
        HNil
      conv2dSpec inChannel outChannel kernelHeight kernelWidth =
        Conv2dSpec' inChannel outChannel kernelHeight kernelWidth (1,1) (0,0)
  in
    conv2dSpec 3 64 3 3 .*.
    conv2dSpec 64 64 3 3 .*.
    maxPool2dSpec .*.
    conv2dSpec 64 128 3 3 .*.
    conv2dSpec 128 128 3 3 .*.
    maxPool2dSpec .*.
    conv2dSpec 128 256 3 3 .*.
    conv2dSpec 256 256 3 3 .*.
    conv2dSpec 256 256 3 3 .*.
    maxPool2dSpec .*.
    conv2dSpec 256 512 3 3 .*.
    conv2dSpec 512 512 3 3 .*.
    conv2dSpec 512 512 3 3 .*.
    AdaptiveAvgPool2dSpec (7,7) .*.
    ReshapeSpec [1,512*7*7] .*.
    vggClassifierSpec
  
-- resnetSpec numClass =
--   conv2dSpec 3 64 7 7 .*.
--   BatchNorm2dSpec 64 .*.
--   ReluSpec .*.
--   conv2dSpec 64 64 3 3 .*.
--   HNil
