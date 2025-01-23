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
{-# LANGUAGE TypeFamilies#-}
{-# LANGUAGE GADTs#-}
{-# LANGUAGE OverloadedRecordDot#-}
{-# LANGUAGE FlexibleContexts#-}
{-# LANGUAGE FunctionalDependencies#-}
{-# LANGUAGE TypeFamilyDependencies#-}
{-# LANGUAGE PartialTypeSignatures #-}


module Torch.Compose where

import Torch
import Torch.NN
import Torch.Functional
import GHC.Generics hiding ((:+:))

data (:>>:) a b = (:>>:)
  { head :: a
  , tail :: b
  } deriving (Show, Eq, Generic)

infixr 5 :>>:

data (://:) a b = Fanout
  { head :: a
  , tail :: b
  } deriving (Show, Eq, Generic)

data (:+:) a b = Fanin
  { head :: a
  , tail :: b
  } deriving (Show, Eq, Generic)

data (:++:) a b = Concat
  { head :: a
  , tail :: b
  } deriving (Show, Eq, Generic)

instance (Randomizable spec0 f0, Randomizable spec1 f1) => Randomizable (spec0 :>>: spec1) (f0 :>>: f1) where
  sample ((:>>:) s0 s1) = do
    f0 <- sample s0
    f1 <- sample s1
    return ((:>>:) f0 f1)

instance (HasForward f a b, HasForward g b c) => HasForward (f :>>: g) a c where
  forward ((:>>:) f g) a = forward g (forward f a)
  forwardStoch ((:>>:) f g) a = forwardStoch f a >>= forwardStoch g

instance (Randomizable spec0 f0, Randomizable spec1 f1) => Randomizable (spec0 ://: spec1) (f0 ://: f1) where
  sample (Fanout s0 s1) = do
    f0 <- sample s0
    f1 <- sample s1
    return (Fanout f0 f1)

instance (HasForward f a b, HasForward g a c) => HasForward (f ://: g) a (b,c) where
  forward (Fanout f g) a = (forward f a, forward g a)
  forwardStoch (Fanout f g) a = (,) <$> forwardStoch f a <*> forwardStoch g a

instance (Randomizable spec0 f0, Randomizable spec1 f1) => Randomizable (spec0 :+: spec1) (f0 :+: f1) where
  sample (Fanin s0 s1) = do
    f0 <- sample s0
    f1 <- sample s1
    return (Fanin f0 f1)

instance (Num c, HasForward f a c, HasForward g b c) => HasForward (f :+: g) (a,b) c where
  forward (Fanin f g) (a,b) = forward f a + forward g b
  forwardStoch (Fanin f g) (a,b) = do
    c <- forwardStoch f a
    c' <- forwardStoch g b
    return (c + c')

instance (Randomizable spec0 f0, Randomizable spec1 f1) => Randomizable (spec0 :++: spec1) (f0 :++: f1) where
  sample (Concat s0 s1) = do
    f0 <- sample s0
    f1 <- sample s1
    return (Concat f0 f1)

instance (HasForward f a0 b0, HasForward g a1 b1) => HasForward (f :++: g) (a0,a1) (b0,b1) where
  forward (Concat f g) (a0,a1) = (forward f a0, forward g a1)
  forwardStoch (Concat f g) (a0,a1) = do
    b0 <- forwardStoch f a0
    b1 <- forwardStoch g a1
    return (b0, b1)

data ShortcutSpec a = ShortcutSpec a deriving (Show, Generic)

data Shortcut a = Shortcut a deriving (Show, Generic, Parameterized)

instance (Randomizable a b) => Randomizable (ShortcutSpec a) (Shortcut b) where
  sample (ShortcutSpec s) = Shortcut <$> sample s

instance (HasForward a Tensor Tensor) => HasForward (Shortcut a) Tensor Tensor where
  forward (Shortcut f) input = forward f input + input
  forwardStoch (Shortcut f) input = do
    f' <- forwardStoch f input
    return $ f' + input

data ReplicateSpec b = ReplicateSpec Int b deriving (Show, Generic)
data Replicate b = Replicate [b] deriving (Show, Generic)

instance (Randomizable b c) => Randomizable (ReplicateSpec b) (Replicate c) where
  sample (ReplicateSpec n s) = Replicate <$> sequence (replicate n (sample s))

instance (HasForward a b b) => HasForward (Replicate a) b b where
  forward (Replicate []) input = input
  forward (Replicate (a:ax)) input = forward (Replicate ax) (forward a input)
  forwardStoch (Replicate []) input = pure input
  forwardStoch (Replicate (a:ax)) input = forwardStoch (Replicate ax) =<< forwardStoch a input

type family LastLayer x where
  LastLayer (a :>>: b) = LastLayer b
  LastLayer x          = x

class HasLast x r | x -> r where
  getLast :: x -> r

instance HasLast b r => HasLast (a :>>: b) r where
  getLast ((:>>:) _ b) = getLast b

instance HasLast a a where
  getLast = id

type family FirstLayer x where
  FirstLayer (a :>>: b) = a
  FirstLayer x          = x

class HasFirst x r | x -> r where
  getFirst :: x -> r

instance HasFirst a r => HasFirst (a :>>: b) r where
  getFirst ((:>>:) a _) = getFirst a

instance HasFirst a a where
  getFirst = id

class HasForwardAssoc f a where
  type ForwardResult f a
  forwardAssoc :: f -> a -> ForwardResult f a

class HasOutputs f a where
  type Outputs f a
  toOutputs :: f -> a -> Outputs f a

instance (HasForwardAssoc f0 a, HasOutputs f0 a, HasOutputs f1 (ForwardResult f0 a)) => HasOutputs (f0 :>>: f1) a where
  type Outputs (f0 :>>: f1) a = Outputs f0 a :>>: Outputs f1 (ForwardResult f0 a)
  toOutputs ((:>>:) f0 f1) a  = (:>>:) (toOutputs f0 a) (toOutputs f1 (forwardAssoc f0 a))

class HasInputs f a where
  type Inputs f a
  toInputs :: f -> a -> Inputs f a

instance (HasForwardAssoc f0 a, HasInputs f0 a, HasInputs f1 (ForwardResult f0 a)) => HasInputs (f0 :>>: f1) a where
  type Inputs (f0 :>>: f1) a = Inputs f0 a :>>: Inputs f1 (ForwardResult f0 a)
  toInputs ((:>>:) f0 f1) a  = (:>>:) (toInputs f0 a) (toInputs f1 (forwardAssoc f0 a))


class HasOutputShapes f a where
  type OutputShapes f a
  toOutputShapes :: f -> a -> OutputShapes f a

instance (HasForwardAssoc f0 a, HasOutputShapes f0 a, HasOutputShapes f1 (ForwardResult f0 a)) => HasOutputShapes (f0 :>>: f1) a where
  type OutputShapes (f0 :>>: f1) a = OutputShapes f0 a :>>: OutputShapes f1 (ForwardResult f0 a)
  toOutputShapes ((:>>:) f0 f1) a  = (:>>:) (toOutputShapes f0 a) (toOutputShapes f1 (forwardAssoc f0 a))

