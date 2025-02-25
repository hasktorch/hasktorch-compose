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
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE RecordWildCards#-}
{-# LANGUAGE ScopedTypeVariables#-}
{-# LANGUAGE StandaloneDeriving#-}
{-# LANGUAGE TypeApplications#-}
{-# LANGUAGE TypeFamilies#-}
{-# LANGUAGE TypeFamilyDependencies#-}
{-# LANGUAGE TypeOperators#-}
{-# LANGUAGE UndecidableInstances#-}
{-# LANGUAGE AllowAmbiguousTypes#-}


module Torch.Compose where

import Torch
import GHC.Generics hiding ((:+:))
import Data.HList
import Data.Kind
import Data.Coerce
import Control.Exception
import System.IO.Unsafe
import qualified Language.Haskell.TH as TH
import GHC.TypeLits
-- import qualified Language.Haskell.TH.Syntax as TH


deriving instance Generic (HList '[])
deriving instance Generic a => Generic (HList (a ': ax))

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
  { first :: a
  , second :: b
  } deriving (Show, Eq, Generic)

data (:+:) a b = Fanin
  { first :: a
  , second :: b
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

newtype ShortcutSpec a = ShortcutSpec a deriving (Show, Generic)

newtype Shortcut a = Shortcut a deriving (Show, Generic, Parameterized)

instance (Randomizable a b) => Randomizable (ShortcutSpec a) (Shortcut b) where
  sample (ShortcutSpec s) = Shortcut <$> sample s

instance (HasForward a Tensor Tensor) => HasForward (Shortcut a) Tensor Tensor where
  forward (Shortcut f) input = forward f input + input
  forwardStoch (Shortcut f) input = do
    f' <- forwardStoch f input
    return $ f' + input

data ReplicateSpec b = ReplicateSpec Int b deriving (Show, Generic)
newtype Replicate b = Replicate [b] deriving (Show, Generic)

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
  forwardStochAssoc :: f -> a -> IO (ForwardResult f a)

toHList :: x -> HList '[x]
toHList x = HCons x HNil

instance (HasForwardAssoc f a) => HasForwardAssoc f (HList '[a]) where
  type ForwardResult f (HList '[a]) = HList '[ForwardResult f a]
  forwardAssoc f (HCons a HNil) = toHList $ forwardAssoc f a
  forwardStochAssoc f (HCons a HNil) = toHList <$> forwardStochAssoc f a

type family ToHNat (x :: Nat) :: HNat where
  ToHNat 0 = HZero
  ToHNat x = HSucc (ToHNat ( x - 1 ))

dropLayers :: forall num a ys xs. (KnownNat num, Coercible a (HList xs), HDrop (ToHNat num) xs ys,  HLengthGe xs (ToHNat num)) => a -> HList ys
dropLayers m = hDrop (Proxy :: Proxy (ToHNat num)) (coerce m)

takeLayers :: forall num a ys xs. (KnownNat num, Coercible a (HList xs), HTake (ToHNat num) xs ys,  SameLength' (HReplicateR (ToHNat num) ()) ys,  HLengthEq1 ys (ToHNat num), HLengthEq2 ys (ToHNat num),  HLengthGe xs (ToHNat num)) => a -> HList ys
takeLayers m = hTake (Proxy :: Proxy (ToHNat num)) (coerce m)

splitLayers :: forall num a xs ys xsys. (KnownNat num, Coercible a (HList xsys), HSplitAt1 '[] (ToHNat num) xsys xs ys,  HAppendList1 xs ys xsys,  SameLength' (HReplicateR (ToHNat num) ()) xs,  HLengthEq1 xs (ToHNat num), HLengthEq2 xs (ToHNat num)) => a -> (HList xs, HList ys)
splitLayers m = hSplitAt (Proxy :: Proxy (ToHNat num)) (coerce m)

dropLastLayer :: (Coercible a (HList xs1), HRevApp xs2 '[x] xs1, HRevApp xs2 '[] sx,  HRevApp xs1 '[] (x : xs2), HRevApp sx '[] xs2) => a -> HList sx
dropLastLayer m = hReverse (hDrop (Proxy :: Proxy (HSucc HZero)) (hReverse (coerce m)))

addLastLayer :: HAppend l1 (HList '[e]) => l1 -> e -> HAppendR l1 (HList '[e])
addLastLayer a b = a `hAppend` (b .*. HNil)

dropFirstLayer :: Coercible a (HList (x : ys)) => a -> HList ys
dropFirstLayer m = hDrop (Proxy :: Proxy (HSucc HZero)) (coerce m)

getLastLayer :: (Coercible a (HList l1), HRevApp l1 '[] (e : l)) => a -> e
getLastLayer a = hLast (coerce a)

getFirstLayer :: Coercible a (HList (e : l)) => a -> e
getFirstLayer a = hHead (coerce a)

hScanl :: forall f z ls xs1 sx xs2. (HScanr f z ls xs1, HRevApp xs1 '[] sx, HRevApp sx '[] xs1,  HRevApp xs2 '[] ls, HRevApp ls '[] xs2) => f -> z -> HList xs2 -> HList sx
hScanl a b c = hReverse $ hScanr a b (hReverse c)

safeEval :: forall a. a -> Maybe a
safeEval x = unsafePerformIO $ do
  result <- try (evaluate @a x) :: IO (Either SomeException a)
  case result of
    Left  _ -> return Nothing
    Right v -> return (Just v)

type family ForwardMap (xs :: [Type]) (a :: Type) :: [Type] where
  ForwardMap '[]  _ = '[]
  ForwardMap (x ': xs) a = ForwardResult x a ': ForwardMap xs (ForwardResult x a)

class Outputs xs input where
  toOutputs' :: HList xs -> input -> HList (ForwardMap xs input)
  toOutputsWithStoch' :: HList xs -> input -> IO (HList (ForwardMap xs input))

instance HasForwardAssoc x a => HasForwardAssoc x (Maybe a) where
  type ForwardResult x (Maybe a) = Maybe (ForwardResult x a)
  forwardAssoc x (Just a) = Just $ forwardAssoc x a
  forwardAssoc x Nothing = Nothing
  forwardStochAssoc x (Just a) = Just <$> forwardStochAssoc x a
  forwardStochAssoc x Nothing = pure Nothing
  

instance (HasForwardAssoc x a, Outputs xs (ForwardResult x a)) => Outputs (x ': xs) a where
  toOutputs' (HCons x xs) a =
    let out = forwardAssoc x a
    in HCons out $ toOutputs' xs out
  toOutputsWithStoch' (HCons x xs) a = do
    out <- forwardStochAssoc x a
    HCons out <$> toOutputsWithStoch' xs out

instance Outputs '[] a where
  toOutputs' _ _ = HNil
  toOutputsWithStoch' _ _ = pure HNil

toOutputs ::
  (Coercible a (HList xs),
   Outputs xs input
  ) =>
  a -> input -> HList (ForwardMap xs input)
toOutputs f = toOutputs' (coerce f)

toOutputsWithStoch ::
  (Coercible a (HList xs),
   Outputs xs input
  ) =>
  a -> input -> IO (HList (ForwardMap xs input))
toOutputsWithStoch f = toOutputsWithStoch' (coerce f)

toOutputShapes ::
  (Coercible a (HList xs),
   HMapAux HList (Tensor -> [Int]) (ForwardMap xs input) b,
   SameLength' b (ForwardMap xs input),
   SameLength' (ForwardMap xs input) b, Outputs xs input
  ) =>
  a -> input -> HList b
toOutputShapes f a = hMap shape (toOutputs f a) 

toOutputShapesWithStoch ::
  (Coercible a (HList xs),
   HMapAux HList (Tensor -> [Int]) (ForwardMap xs input) b,
   SameLength' b (ForwardMap xs input),
   SameLength' (ForwardMap xs input) b, Outputs xs input
  ) =>
  a -> input -> IO (HList b)
toOutputShapesWithStoch f a = hMap shape <$> toOutputsWithStoch f a

toMaybeOutputShapes ::
  (Coercible a (HList xs),
   HMapAux HList (Tensor -> Maybe [Int]) (ForwardMap xs input) b,
   SameLength' b (ForwardMap xs input),
   SameLength' (ForwardMap xs input) b, Outputs xs input
  ) =>
  a -> input -> HList b
toMaybeOutputShapes f a = hMap (safeEval . shape) (toOutputs f a) 

toMaybeOutputShapesWithStoch ::
  (Coercible a (HList xs),
   HMapAux HList (Tensor -> Maybe [Int]) (ForwardMap xs input) b,
   SameLength' b (ForwardMap xs input),
   SameLength' (ForwardMap xs input) b, Outputs xs input
  ) =>
  a -> input -> IO (HList b)
toMaybeOutputShapesWithStoch f a = hMap (safeEval . shape) <$> toOutputsWithStoch f a

mergeParameters :: Parameterized a => (Tensor -> Tensor -> Tensor) -> a -> a -> a
mergeParameters fn a b =
  replaceParameters b $
    zipWith
      (\a' b' -> IndependentTensor $ fn (toDependent a') (toDependent b'))
      (flattenParameters a)
      (flattenParameters b)

instanceForwardAssoc :: TH.Q TH.Type -> TH.Q TH.Type -> TH.Q TH.Type -> TH.DecsQ
instanceForwardAssoc model input output =
  [d|
      instance HasForwardAssoc $model $input where
        type ForwardResult $model $input = $output
        forwardAssoc = forward
        forwardStochAssoc = forwardStoch
  |]

instanceForwardAssocs :: [TH.Q TH.Type] -> TH.Q TH.Type -> TH.Q TH.Type -> TH.DecsQ
instanceForwardAssocs models input output =
  concat <$> forM models (\model -> instanceForwardAssoc model input output)
