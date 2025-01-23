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

module Torch.Compose.Models where

import Torch
import Torch.Compose
import Torch.Compose.NN
import GHC.Generics hiding ((:+:))

vgg16Spec numClass =
  let maxPool2dSpec = MaxPool2dSpec
        { kernelSize = (3,3)
        , stride = (2,2)
        , padding = (1,1)
        , dilation = (0,0)
        , ceilMode = Ceil
        }
      vggClassifierSpec =
        LinearSpec (512 * 7 * 7) 4096 :>>:
        ReluSpec :>>:
        DropoutSpec 0.5 :>>:
        LinearSpec 4096 4096 :>>:
        ReluSpec :>>:
        DropoutSpec 0.5 :>>:
        LinearSpec 4096 numClass
  in
    Conv2dSpec 3 64 3 3 :>>:
    Conv2dSpec 64 64 3 3 :>>:
    maxPool2dSpec :>>:
    Conv2dSpec 64 128 3 3 :>>:
    Conv2dSpec 128 128 3 3 :>>:
    maxPool2dSpec :>>:
    Conv2dSpec 128 256 3 3 :>>:
    Conv2dSpec 256 256 3 3 :>>:
    Conv2dSpec 256 256 3 3 :>>:
    maxPool2dSpec :>>:
    Conv2dSpec 256 512 3 3 :>>:
    Conv2dSpec 512 512 3 3 :>>:
    Conv2dSpec 512 512 3 3 :>>:
    AdaptiveAvgPool2dSpec (7,7) :>>:
    ReshapeSpec [1,512*7*7] :>>:
    vggClassifierSpec
  

resnetSpec numClass =
  Conv2dSpec 3 64 7 7 :>>:
  BatchNorm2dSpec 64 :>>:
  ReluSpec :>>:
  Conv2dSpec 64 64 3 3
