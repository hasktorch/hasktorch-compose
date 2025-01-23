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
        Forward (LinearSpec (512 * 7 * 7) 4096) $
        Forward ReluSpec $
        Forward (DropoutSpec 0.5) $
        Forward (LinearSpec 4096 4096) $
        Forward ReluSpec $
        Forward (DropoutSpec 0.5) $
        (LinearSpec 4096 numClass)
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
  vggClassifierSpec
