{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveAnyClass#-}
{-# LANGUAGE DeriveGeneric#-}
{-# LANGUAGE DuplicateRecordFields#-}
{-# LANGUAGE FlexibleContexts#-}
{-# LANGUAGE FlexibleInstances#-}
{-# LANGUAGE FunctionalDependencies#-}
{-# LANGUAGE GADTs#-}
{-# LANGUAGE MultiParamTypeClasses#-}
{-# LANGUAGE OverloadedRecordDot#-}
{-# LANGUAGE PartialTypeSignatures #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RecordWildCards#-}
{-# LANGUAGE ScopedTypeVariables#-}
{-# LANGUAGE TypeApplications#-}
{-# LANGUAGE TypeFamilies#-}
{-# LANGUAGE TypeFamilyDependencies#-}
{-# LANGUAGE TypeOperators#-}
{-# LANGUAGE UndecidableInstances#-}


module Torch.Compose where

import Torch
import Torch.NN
import Torch.Functional
import GHC.Generics hiding ((:+:))
-- import Data.Void
import Data.HList
import Data.HList (hAppend)
import Data.Kind
import Data.Coerce
import Control.Exception
import System.IO.Unsafe

instance (Randomizable spec0 f0, Randomizable (HList spec1) (HList f1)) => Randomizable (HList (spec0 ': spec1)) (HList (f0 ': f1)) where
  sample (HCons s0 s1) = do
    f0 <- sample s0
    f1 <- sample s1
    return (f0 .*. f1)

instance Randomizable (HList '[]) (HList '[]) where
  sample HNil = do
    return HNil

instance (HasForward f a b, HasForward (HList g) b c) => HasForward (HList (f ': g)) a c where
  forward (HCons f g) a = forward g (forward f a)
  forwardStoch (HCons f g) a = forwardStoch f a >>= forwardStoch g

instance HasForward (HList '[]) a a where
  forward _ = id
  forwardStoch _ = pure

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

class HasForwardAssoc f a where
  type ForwardResult f a :: Type
  forwardAssoc :: f -> a -> ForwardResult f a

toHList :: x -> HList '[x]
toHList x = HCons x HNil

instance (HasForwardAssoc f a) => HasForwardAssoc f (HList '[a]) where
  type ForwardResult f (HList '[a]) = HList '[ForwardResult f a]
  forwardAssoc f (HCons a HNil) = toHList $ forwardAssoc f a

dropLastLayer :: (Coercible a (HList xs1), HRevApp xs2 '[x] xs1, HRevApp xs2 '[] sx,  HRevApp xs1 '[] (x : xs2), HRevApp sx '[] xs2) => a -> HList sx
dropLastLayer m = hReverse (hDrop (Proxy :: Proxy (HSucc HZero)) (hReverse (coerce m)))

addLastLayer :: HAppend l1 (HList '[e]) => l1 -> e -> HAppendR l1 (HList '[e])
addLastLayer a b = a `hAppend` (b .*. HNil)

getLastLayer :: (Coercible a (HList l1), HRevApp l1 '[] (e : l)) => a -> e
getLastLayer a = hLast (coerce a)

hScanl :: forall f z ls xs1 sx xs2. (HScanr f z ls xs1, HRevApp xs1 '[] sx, HRevApp sx '[] xs1,  HRevApp xs2 '[] ls, HRevApp ls '[] xs2) => f -> z -> HList xs2 -> HList sx
hScanl a b c = hReverse $ hScanr a b (hReverse c)

safeEval :: forall a. a -> Maybe a
safeEval x = unsafePerformIO $ do
  result <- try (evaluate @a x) :: IO (Either SomeException a)
  case result of
    Left  _ -> return Nothing
    Right v -> return (Just v)

type family ForwardMap (xs :: [*]) (a :: *) :: [*] where
  ForwardMap '[]  _ = '[]
  ForwardMap (x ': xs) a = ForwardResult x a ': ForwardMap xs (ForwardResult x a)

class Outputs xs input where
  toOutputs' :: HList xs -> input -> HList (ForwardMap xs input)

instance HasForwardAssoc x a => HasForwardAssoc x (Maybe a) where
  type ForwardResult x (Maybe a) = Maybe (ForwardResult x a)
  forwardAssoc x (Just a) = Just $ forwardAssoc x a
  forwardAssoc x Nothing = Nothing
  

instance (HasForwardAssoc x a, Outputs xs (ForwardResult x a)) => Outputs (x ': xs) a where
  toOutputs' (HCons x xs) a =
    let out = forwardAssoc x a
    in HCons out $ toOutputs' xs out

instance Outputs '[] a where
  toOutputs' _ _ = HNil

toOutputs ::
  (Coercible a (HList xs),
   Outputs xs input
  ) =>
  a -> input -> HList (ForwardMap xs input)
toOutputs f = toOutputs' (coerce f)

toOutputShapes ::
  (Coercible a (HList xs),
   HMapAux HList (Tensor -> [Int]) (ForwardMap xs input) b,
   SameLength' b (ForwardMap xs input),
   SameLength' (ForwardMap xs input) b, Outputs xs input
  ) =>
  a -> input -> HList b
toOutputShapes f a = hMap shape (toOutputs f a) 

toMaybeOutputShapes ::
  (Coercible a (HList xs),
   HMapAux HList (Tensor -> Maybe [Int]) (ForwardMap xs input) b,
   SameLength' b (ForwardMap xs input),
   SameLength' (ForwardMap xs input) b, Outputs xs input
  ) =>
  a -> input -> HList b
toMaybeOutputShapes f a = hMap (safeEval . shape) (toOutputs f a) 
