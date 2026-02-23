import torch
import numpy as np


class BatchTokenCharacterTrie:
    r"""A GPU-optimized version of TokenCharacterTrie that performs mass_sum in parallel.

    The mass at leaf nodes is propagated to their ancestors through sparse matrix
    multiplication with the [transitive] reachability matrix.

    The reachability matrix M is a num_leafs × num_nodes matrix
    where M[i,j] = 1 if:
        - leaf_indices[i] == j (self connection) or
        - j is an ancestor of leaf_indices[i] in the trie

    Example:
        Trie:          M:
              0           [[1, 1, 0, 1],
             / \           [1, 0, 1, 0]]
            1   2 (1)
            |
            3 (0)

    The matrix is stored as a sparse tensor in CSR (Compressed Sparse Row) format,
    built from COO (Coordinate) format. For example,
        rows = [1, 0, 1, 0, 0] (index of leaf node)
        cols = [2, 3, 0, 1, 0] (connections)
        vals = [1, 1, 1, 1, 1] (connection weights)

    When computing masses (batch_size × num_leafs) @ M, each leaf node's mass
    flows up to all its ancestors.
    """


    def __init__(self, words, encode, old_eos, new_eos, device=None):

        # TODO: Perhaps `TokenCharacterTrie` should wrap a TokenizedLLM.
        # TODO: I think that the information about the vocabulary that we see
        # above could be encapsulated better.
        use_bytes = isinstance(words[0], bytes)
        if use_bytes:
            if not isinstance(old_eos, bytes):
                old_eos = old_eos.encode('utf-8')
            if not isinstance(new_eos, bytes):
                new_eos = new_eos.encode('utf-8')

        self.old_eos = old_eos
        self.old_eos_id = encode[old_eos]
        self.new_eos = new_eos

        word2leaf = {}
        children = {}
        root = 0
        children = [{}]
        token_id_to_leaf = []

        for word in words:
            # coerce old eos to new eos
            _word = word
            if word == self.old_eos:
                word = self.new_eos

            curr = root
            for letter in word:
                if letter not in children[curr]:
                    children[curr][letter] = len(children)
                    children.append({})
                curr = children[curr][letter]

            children[curr][None] = last = len(children)
            children.append({})
            word2leaf[word] = last

            token_id_to_leaf.append((encode[_word], last))

        self.token_id_to_leaf = np.array(token_id_to_leaf)
        self.root = root
        self.children = children
        self.word2leaf = word2leaf
        self.leaf2word = dict(zip(self.word2leaf.values(), self.word2leaf.keys()))

        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.M = self._build_reachability_matrix()
        self.token_ids = torch.LongTensor(self.token_id_to_leaf[:, 0], device=self.device)

    def _build_reachability_matrix(self):
        rows, cols = [], []

        leaf_indices = self.token_id_to_leaf[:, 1]

        parent = {}
        for node in range(len(self.children)):
            for child in self.children[node].values():
                parent[child] = node

        for i, node in enumerate(leaf_indices):

            # add self connections
            rows.append(i)
            cols.append(node)

            # add all ancestor connections
            current = node
            while current in parent:        # Walk up to root
                ancestor = parent[current]
                rows.append(i)
                cols.append(ancestor)
                current = ancestor

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.ones(len(rows), device=self.device)
        return torch.sparse_coo_tensor(
            indices, values, (len(leaf_indices), len(self.children))
        ).to_sparse_csr()

    def batch_mass_sum(self, p_llms):
        return torch.sparse.mm(p_llms[:, self.token_ids], self.M)   # pylint: disable=E1102


def test_batched_vs_sequential():
    import gc
    from tokenization.trie import TokenCharacterTrie
    from tokenization.lm import LazyProb
    #from arsenal.maths import compare
    try:
        from tokenization.vllm import load_model_by_name
    except ImportError:
        from tokenization.lm import load_model_by_name


    llm = load_model_by_name('gpt2')

    if not hasattr(llm, 'batch_logp_next'):
        def batch_logp_next(contexts, keep_on_gpu):
            return torch.Tensor([llm.logp_next(context)._p for context in contexts])
        llm.batch_logp_next = batch_logp_next


    words = llm._decode
    encode = llm._encode
    old_eos = '\n' # test non-canonical EOS
    new_eos = '▪'

    sequential_trie = TokenCharacterTrie(words, encode, old_eos, new_eos)
    batch_trie = BatchTokenCharacterTrie(words, encode, old_eos, new_eos)

    contexts = [
        (((), 'Once'), ' the'),
        (((), 'Once'), ' he'),
        ((), 'Once'),
    ]
    p_llms = llm.batch_logp_next(contexts, keep_on_gpu=True).exp()

    batch_results = batch_trie.batch_mass_sum(p_llms)
    sequential_results = torch.zeros_like(batch_results)

    def traverse(A, B, i, j):
        assert abs(A[i] - B[j]) / abs(A[i]) < 0.001, [A[i], B[j], i, j]
        assert sequential_trie.children[i].keys() == batch_trie.children[j].keys()
        for a in sequential_trie.children[i]:
            traverse(A, B, sequential_trie.children[i][a], batch_trie.children[j][a])

    for i in range(len(p_llms)):
        p_llm = p_llms[i].cpu().numpy()
        sequential_results[i] = torch.from_numpy(
            sequential_trie.mass_sum(LazyProb(p_llm, encode, words))
        )

        traverse(
            sequential_results[i].cpu(),
            batch_results[i,:].cpu(),
            sequential_trie.root,
            batch_trie.root,
        )

        # check EOS
        have = batch_results[i, batch_trie.word2leaf[batch_trie.new_eos]].item()
        want = p_llm[encode[batch_trie.old_eos]]
        assert abs(have - want) / abs(want) <= 0.001, [have, want]

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    from arsenal import testing_framework
    testing_framework(globals())
