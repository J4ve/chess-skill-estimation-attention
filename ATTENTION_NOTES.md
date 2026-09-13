# Attention Module — Design Notes for Defense

## Scoring form

The `BahdanauAttention` module in this prototype implements **full additive
(query-key) attention** as defined by Bahdanau et al. (2015):

    e(t, s) = v^T · tanh(W_q · h_t + W_k · h_s)

where h_t is the BiLSTM output at query position t, h_s is the output at key
position s, W_q and W_k are learned projection matrices, and v is a learned
scoring vector.  This is the textbook formulation: the relevance of key s
*depends on* the query position t, so a move's importance can shift depending
on where in the game the model is currently predicting.  The module was
initially implemented with a key-only variant (scores independent of query
position), which is a valid content-based attention form used in some
seq2seq work, but does not match the Bahdanau formulation the thesis claims.
The upgrade to full query-key scoring adds one linear projection (W_q) and
negligible compute (one (T, T, d) broadcast-add per game, where T ≤ 100 and
d = 64), while making the "Bahdanau-style additive attention" claim in
Chapter 3 fully honest.  The causal-cumulative masking is preserved: at each
ply t, attention is restricted to keys s ≤ t within the valid (non-padded)
sequence, so per-move output is maintained and no lookahead is introduced.
