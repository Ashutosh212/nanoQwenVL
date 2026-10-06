# Small Qwen-style language model — independent exercises

The merger produces visual tokens with width 96. The language model therefore
also uses hidden width 96 so image and text vectors can eventually occupy one
decoder sequence.

## Roadmap

1. token embedding lookup;
2. causal attention mask;
3. one-dimensional text RoPE;
4. grouped-query causal self-attention;
5. RMSNorm and SwiGLU decoder block;
6. decoder stack and vocabulary logits;
7. shifted next-token cross-entropy loss;
8. visual-token insertion at image placeholder positions.

A tokenizer is separate from the neural network. It converts text strings into
integer token IDs. Exercise 1 begins with explicit IDs so the tensor operation
inside the model remains visible.

## Exercise 1: token embedding

Open:

```text
src/qwen25_scratch/language_steps/token_embedding.py
```

Run:

```bash
cd /sfs/qwen2.5
source .venv/bin/activate
python -m qwen25_scratch.language_steps.token_embedding
```

The supplied input and expected output are:

```text
token IDs:       [B,L]   = [2,6]
embedding table: [V,D]   = [128,96]
embeddings:      [B,L,D] = [2,6,96]
```

For an ID `i`, embedding lookup selects row `i`:

```text
output[b, position, :] = weight[token_ids[b, position], :]
```

Therefore every occurrence of token ID 11 returns the same vector before any
attention or positional encoding is applied. During backward, only vocabulary
rows used by the batch receive gradients.

Complete TODO A in the constructor first, then TODO B in `forward`. The
validator checks shapes, repeated-token equality, valid gradients, and that an
unused vocabulary row receives zero gradient.

## Exercise 2: causal attention mask

Open:

```text
src/qwen25_scratch/language_steps/causal_mask.py
```

Run:

```bash
python -m qwen25_scratch.language_steps.causal_mask
```

For sequence length five, rows are query positions and columns are key
positions. Query `q` may see key `k` exactly when `k <= q`:

```text
[[1,0,0,0,0],
 [1,1,0,0,0],
 [1,1,1,0,0],
 [1,1,1,1,0],
 [1,1,1,1,1]]
```

You will build both:

- a boolean `[L,L]` mask for inspecting the rule;
- an additive `[1,1,L,L]` mask containing zero at permitted positions and
  negative infinity at future positions.

The singleton batch and head axes allow this mask to broadcast over attention
scores shaped `[B,A,L,L]`. Adding negative infinity before softmax makes every
future-token probability exactly zero.

## Exercise 3: one-dimensional text RoPE

Open:

```text
src/qwen25_scratch/language_steps/text_rope.py
```

Run:

```bash
python -m qwen25_scratch.language_steps.text_rope
```

Text has one position axis, so position IDs are simply `[0,1,...,L-1]`.
With the learning configuration:

```text
hidden size:       96
query heads:        4
head dimension:    24
inverse frequencies: 12
```

Complete three sections:

1. construct and register `inv_freq [12]`;
2. produce `cos/sin [1,1,L,24]`;
3. apply the rotation to queries and keys `[B,A,L,24]`.

As in vision RoPE, position zero is unchanged and every rotated vector keeps
its L2 norm. Unlike vision RoPE, there is no height/width split because text has
only one ordered sequence axis. Values are not rotated.

## Exercise 4: multi-head causal self-attention

Open:

```text
src/qwen25_scratch/language_steps/multi_head_attention.py
```

Run:

```bash
python -m qwen25_scratch.language_steps.multi_head_attention
```

This exercise connects the three previous language-model pieces using standard
multi-head attention. Queries, keys, and values all use four heads:

```text
input hidden states: [B,L,96]
Q, K, and V:         [B,4,L,24]
attention scores:    [B,4,L,L]
output:              [B,L,96]
```

Each attention head therefore owns one query head, one key head, and one value
head. There is no head repetition in this exercise. The
`num_key_value_heads` setting is reserved for a later grouped-query attention
lesson. Complete TODO A to construct the layers, then follow TODO B one tensor
transformation at a time. The validator checks shapes, exact causal masking,
probability rows, gradient flow, and that changing the last token cannot affect
any earlier output.

## Exercise 5: Qwen-style SwiGLU MLP

Open:

```text
src/qwen25_scratch/language_steps/language_mlp.py
```

Run:

```bash
python -m qwen25_scratch.language_steps.language_mlp
```

The feed-forward network processes each token independently. It expands the
96-dimensional token through two learned paths, gates one path with SiLU, then
projects back to the model dimension:

```text
x:                    [B,L,96]
gate(x), up(x):        [B,L,256]
SiLU(gate(x)) * up(x): [B,L,256]
down(...):             [B,L,96]
```

Unlike attention, this module does not mix information between token
positions. Complete the three projections in TODO A, then implement the SwiGLU
equation one line at a time in TODO B.
