# WARNING: Activation Decoding doesn't work nicely with newer version of transformers, so commenting out a large portion of the main code
# from .baseline import Baseline
# from .baseline_masked_retrieval_head import BaselineMaskedRetrievalHead
# from .baseline_masked_non_retrieval_head import BaselineMaskedNonRetrievalHead
# from .contrastive_decoding import ContrastiveDecoding
# from .decore_vanilla import DeCoReVanilla
# from .decore_entropy import DeCoReEntropy
# from .decore_entropy_gain import DeCoReEntropyGain
# from .dola import DoLa
# from .decore_random_entropy import DeCoReRandomEntropy
from .activation_decoding import ActivationDecoding
