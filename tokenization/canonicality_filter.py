import numpy as np
import scipy.sparse as sp
from tokenization.util import logsumexp
from arsenal.maths import sample as draw


VERYLARGE = 2147483647

class FastCanonicalityFilterBPE:

    def __init__(
            self, _merges, _encode, _decode, _encode_byte, eos_token_id,
            _left, _right, _noncanonical_token_ids, overrides = (),
    ):
        self._encode_byte = _encode_byte
        self._noncanonical_token_ids = list(_noncanonical_token_ids)

        self._left = _left
        self._right = _right

        self._merges = _merges
        self._encode = _encode
        self._decode = _decode
        self.V = len(_decode)          # token vocabulary size

        self.__left_spine, max_left_spine_width = self._left_spine_table()
        self.__right_spine, max_right_spine_width = self._right_spine_table()

        self.left_spine_vector = self.vectorize_spine(self.__left_spine, max_left_spine_width)
        self.right_spine_vector = self.vectorize_spine(self.__right_spine, max_right_spine_width)

        self.indices = np.array([(index, j) for index in range(self.V)
                                 for j in range(len(self.__left_spine[index])-1)])

        self.vector_r = self.left_spine_vector[self.indices[:,0], self.indices[:,1]]
        self.vector_rp = self.left_spine_vector[self.indices[:,0], self.indices[:,1]+1]

        tmp = sp.dok_matrix((self.V, self.V), dtype=np.int32)
        for u, v, uv in _merges:
            tmp[u, v] = uv+1 # +1 to avoid zero-indexing

        self.parent_l_matrix = tmp.tocsr()
        self.parent_l_matrix = self.parent_l_matrix[:, self.vector_r]

        self.eos_token_id = eos_token_id
        self.overrides = dict(overrides)

        for x in overrides:
            assert isinstance(x, int)
            assert isinstance(overrides[x], list)
            assert all(isinstance(y, int) for y in overrides[x])

    def __call__(self, context):
        if context == ():
            mask = np.ones(self.V, dtype=bool)
        else:
            (_, last_token) = context
            left = self._encode[last_token]

            spine_left = self.__right_spine[left]

            L = len(spine_left) - 1    # inf padding

            mask = np.ones(self.V, dtype=bool)

            np_matrix = self.parent_l_matrix[spine_left[:L]].toarray()

            for i in range(L):
                lp = spine_left[i+1]

                vector_k = np_matrix[i]
                # convert 0 in vector_k to VERYLARGE
                vector_k = np.where(vector_k != 0, vector_k-1, VERYLARGE)

                conflict_mask = (vector_k < VERYLARGE)
                conflict_mask &= (vector_k <= self.vector_rp)
                conflict_mask &= (vector_k < lp)
                mask[self.indices[conflict_mask][:,0]] = False

            if left in self.overrides:
                mask[self.overrides[left]] = True

        mask[self._noncanonical_token_ids] = False
        mask[self.eos_token_id] = True

        return mask

    def stable_sample(self, context, logp_next):
        "Numerically stable sampling"
        mask = self(context)
        C = logp_next._p.copy()
        C[~mask] = -np.inf

        N = logp_next._p.copy()
        N[mask] = -np.inf

        N_Z = logsumexp(N)

        if np.log(np.random.uniform()) <= N_Z:
            N_p = np.exp(N - N_Z)
            a = draw(N_p)
        else:
            C_Z = logsumexp(C)
            C_p = np.exp(C - C_Z)
            a = draw(C_p)

        return logp_next._decode[a]

    def vectorize_spine(self, spine, max_spine_width):
        new_spine = [
            [s[i] if i < len(s) else -VERYLARGE for i in range(max_spine_width)]
            for s in spine
        ]
        return np.array(new_spine, dtype=np.int32)

    def _left_spine_table(self):
        "Closure of the left tables."
        max_width = 0
        left_spine = [None]*self.V
        left = self._left
        for i in range(self.V):
            spine = [VERYLARGE, i]
            x = i
            while True:
                x = left[x]
                if x is None: break
                spine.append(x)
            spine.reverse()
            left_spine[i] = spine
            max_width = max(max_width, len(spine))
        return left_spine, max_width

    def _right_spine_table(self):
        "Closure of the right tables."
        max_width = 0
        right_spine = [None]*self.V
        right = self._right
        for i in range(self.V):
            spine = [VERYLARGE, i]
            x = i
            while True:
                x = right[x]
                if x is None: break
                spine.append(x)
            spine.reverse()
            right_spine[i] = spine
            max_width = max(max_width, len(spine))
        return right_spine, max_width

#    def _vectorized_conflicting_next_tokens(self, left: int):
#        spine_left = self.__right_spine[left]
#        L = len(spine_left) - 1    # inf padding
#        conflicts = set()
#        np_matrix = self.parent_l_matrix[[spine_left[i] for i in range(L)]].toarray()
#        for i in range(L):
#            lp = spine_left[i+1]
#            vector_k = np_matrix[i]
#            # convert 0 in vector_k to VERYLARGE
#            vector_k = np.where(vector_k != 0, vector_k-1, VERYLARGE)
#            conflict_mask = (vector_k < VERYLARGE)
#            conflict_mask &= (vector_k <= self.vector_rp)
#            conflict_mask &= (vector_k < lp)
#            conflicts.update(self.indices[conflict_mask][:,0])
#        return conflicts

#    def _vectorized_conflicting_next_tokens2(self, left: int):
#        spine_left = self.__right_spine[left]
#        L = len(spine_left) - 1    # inf padding
#        mask = np.ones(self.V, dtype=bool)
#        np_matrix = self.parent_l_matrix[spine_left[:L]].toarray()
#        for i in range(L):
#            lp = spine_left[i+1]
#            vector_k = np_matrix[i]
#            # convert 0 in vector_k to VERYLARGE
#            vector_k = np.where(vector_k != 0, vector_k-1, VERYLARGE)
#            conflict_mask = (vector_k < VERYLARGE)
#            conflict_mask &= (vector_k <= self.vector_rp)
#            conflict_mask &= (vector_k < lp)
#            mask[self.indices[conflict_mask][:,0]] = False
#        return mask
