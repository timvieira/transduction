import numpy as np
import pandas as pd
import difflib
from collections import Counter, defaultdict
from arsenal import iterview, colors
from tokenization.util import escape, format_alignment, display_table, MyTree


class Inspect:
    """
    Given a collection of `samples`, compute the number of violations as bigram statistics.
    """
    def __init__(self, samples, method, bpe, progress=True):
        self.errors = defaultdict(list)
        self.count = Counter()
        self.method = method
        for token_ids in (iterview(samples) if progress else samples):
            self.add_sample(token_ids)
        self.bpe = bpe            # TODO: bpe attribute is only used in analysis

    def add_sample(self, token_ids):
        T = len(token_ids)
        for t in range(T):
            if not self.method(token_ids[:t], token_ids[t]):    # apply the canonicality test
                pair = (token_ids[t-1], token_ids[t])
                self.errors[pair].append([t, token_ids])
                self.count[pair] += 1

    def analyze(self, top=None, window=2, num_examples=3):
        count = self.count
        errors = self.errors
        bpe = self.bpe
        if len(count) == 0:
            print(colors.green % 'no errors!', colors.trophy)
        for pair, c in count.most_common(top):
            (prev, curr) = pair
            print(c, [bpe._decode[prev], bpe._decode[curr]], pair)
            for t, token_ids in errors[pair][:num_examples]:           # consider randomizing
                hs = bpe.token_ids_to_byte_chunks(token_ids)

                context = hs[t-window-1:t-1] + [hs[t-1], hs[t]] + hs[t+1:t+window+1]
                canonical_context = bpe.canonicalize_byte_chunks(context)

                #assert bpe.canonicalize_byte_chunks([hs[t-1], hs[t]]) != [hs[t-1], hs[t]]
                #assert bpe.canonicalize_token_ids([token_ids[t-1], token_ids[t]]) != [token_ids[t-1], token_ids[t]]

                s1, s2 = format_alignment(
                    tuple(escape(t) for t in context),            # hugging face's tokenization
                    tuple(escape(t) for t in canonical_context),  # our tokenization
                )
                print(colors.bold % '  - context:     ', s1)
                print(colors.bold % '    canonicalize:', s2)


class compare_tokenizers:
    def __init__(self, sentences, f, g, verbosity=0):
        data = []
        for x in iterview(sentences):
            fx = f(x)
            gx = g(x)
            data.append(dict(x=x, fx=fx, gx=gx, agree=(fx == gx)))
            if verbosity > 0 and fx != gx:
                s1, s2 = format_alignment(
                    [escape(t) for t in fx],
                    [escape(t) for t in gx],
                )
                print(colors.bold % 'f:', s1)
                print(colors.bold % 'g:', s2)
        self.df = pd.DataFrame(data)

    def show(self, filter=None, take=np.inf):
        n = 0
        df = self.df[filter] if filter is not None else self.df
        for _, row in df.iterrows():
            n += 1
            if n > take:
                break
            s1, s2 = format_alignment(
                [escape(t) for t in row.fx],
                [escape(t) for t in row.gx],
            )
            print(colors.bold % 'x:', row.x)
            print(colors.bold % 'f:', s1)
            print(colors.bold % 'g:', s2)
            print()

    def show_edits(self, top=None):
        rules = self.identify_replacement_rule(
            [x for xs in self.df.fx for x in xs],
            [x for xs in self.df.gx for x in xs],
        )
        for (old, new), count in rules.most_common(top):
            print(colors.orange % f'{count}:', fmt(old), colors.orange % '==>', fmt(new))

    @staticmethod
    def identify_replacement_rule(sequence1, sequence2):
        "Identifies replacement rules to transform sequence1 into sequence2 using the longest common subsequence alignment."
        # sequence alignment
        matcher = difflib.SequenceMatcher(None, sequence1, sequence2)

        rules = []  # To store the identified rules

        # Process the differences
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                # Append the differing subsequences as a replacement rule
                rules.append((tuple(sequence1[i1:i2]), tuple(sequence2[j1:j2])))
            elif tag == "delete":
                # Append deletion as replacing a subsequence with nothing
                rules.append((tuple(sequence1[i1:i2]), ()))
            elif tag == "insert":
                # Append insertion as replacing nothing with a subsequence
                rules.append(((), tuple(sequence2[j1:j2])))

        return Counter(rules)

    def evaluate_rule_with_mistakes(self, old_pattern, new_pattern):
        """
        Evaluates the quality of a single replacement rule using LCS alignment and identifies mistakes.

        Args:
            rule (tuple): A tuple (old_pattern, new_pattern) representing the rule.
            sequence1 (list): The input sequence from system 1.
            sequence2 (list): The ground truth output sequence from system 2.

        Returns:
            dict: A dictionary containing precision, recall, F1-score, error rate, and lists of mistakes.
        """

        tp = 0
        fp = 0
        n1 = 0
        fn = 0
        for sequence1, sequence2 in zip(self.df.fx, self.df.gx):

            n = len(old_pattern)

            # Apply the rule to generate predicted_sequence2
            predicted_sequence2 = []
            i = 0
            while i < len(sequence1):
                if sequence1[i:i+n] == list(old_pattern):
                    predicted_sequence2.extend(new_pattern)
                    i += n  # Skip the matched pattern
                else:
                    predicted_sequence2.append(sequence1[i])
                    i += 1

            # Align predicted_sequence2 and sequence2 using LCS
            matcher = difflib.SequenceMatcher(None, predicted_sequence2, sequence2)
            matches = matcher.get_matching_blocks()

            # Identify mistakes

            # TODO: we should only count the error if it involves the rule
            fmt1 = []
            fmt2 = []
            has_error = False
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                xs = predicted_sequence2[i1:i2]
                ys = sequence2[j1:j2]
                if tag == "replace":
                    has_error = True
                    fmt1.extend((colors.light.red % escape(x)) for x in xs)
                    fmt2.extend((colors.light.red % escape(x)) for x in ys)
                elif tag == "delete":
                    has_error = True
                    fmt1.extend((colors.light.red % escape(x)) for x in xs)
                elif tag == "insert":
                    has_error = True
                    fmt2.extend((colors.light.red % escape(x)) for x in ys)
                else:
                    fmt1.extend((colors.green % escape(x)) for x in xs)
                    fmt2.extend((colors.green % escape(x)) for x in ys)

            if has_error:
                print(colors.bold % 'original:')
                print((colors.orange % '|').join(escape(x) for x in sequence1))
                print(colors.bold % 'edited:')
                print((colors.orange % '|').join(fmt1))
                print(colors.bold % 'target:')
                print((colors.orange % '|').join(fmt2))
                print()

            # Count True Positives, False Positives, and False Negatives
            Tp = sum(block.size for block in matches)
            tp += Tp  # Total aligned matches
            fp += len(predicted_sequence2) - Tp         # Extra elements in predicted_sequence2
            fn += len(sequence2) - Tp                   # Missing elements in predicted_sequence2
            n1 += len(sequence1)

        # Compute precision, recall, F1, and error rate
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        error_rate = fp / n1 if n1 > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "error_rate": error_rate
        }


def fmt(bs):
    return (colors.orange % '|').join(escape(b) for b in bs)



def Yield(t):
    if isinstance(t, MyTree):
        left, right = t
        return Yield(left) + Yield(right)
    else:
        assert isinstance(t, bytes), t
        return t


def enumerate_trees(self, x):
    assert isinstance(x, bytes)
    if len(x) == 1:
        yield x
    for i in range(1, len(x)):
        for left in enumerate_trees(self, x[:i]):
            for right in enumerate_trees(self, x[i:]):
                if (self._encode[Yield(left)], self._encode[Yield(right)]) in self._parent:
                    yield MyTree(left, right)


def _find_conflict(self, left, right):
    "Search for a conflicting tree with (left, right)."
    l = left
    L = np.inf
    while True:
        r = right
        R = np.inf
        while True:
            k = self._parent.get((self._encode[Yield(l)], self._encode[Yield(r)]), np.inf)      # possible merge of l and r
            if k <= R and k < L:
                # Note: The condition `k < lp` is a strict inequality (unlike `k <= rp`) so
                # that the duplicate merge (`lp == k`) associates left as does in BPE.
                return MyTree(l, r)
            if not isinstance(r, MyTree): break
            R = self._parent[self._encode[Yield(r.left)], self._encode[Yield(r.right)]]
            r = r.left
        if not isinstance(l, MyTree): break
        L = self._parent[self._encode[Yield(l.left)], self._encode[Yield(l.right)]]
        l = l.right


def analyze_conflicts(ex, prev_token, next_token):
    display_table([ex.canonicalize_byte_chunks([prev_token, next_token])])
    rows = []
    for t1 in enumerate_trees(ex, prev_token):
        for t2 in enumerate_trees(ex, next_token):
            rows.append([t1, t2, _find_conflict(ex, t1, t2)])
    display_table(rows, headings=['left', 'right', 'conflict'])


#def new_canonical_test(ex, prev_token, next_token):
#    assert isinstance(prev_token, bytes) and isinstance(next_token, bytes)
#    for t1 in enumerate_trees(ex, prev_token):
#        for t2 in enumerate_trees(ex, next_token):
#            if _find_conflict(ex, t1, t2) is None:
#                return True
#    return False


#def new_canonical_test(ex, prev_token, next_token, fast=True):
#    assert isinstance(prev_token, bytes) and isinstance(next_token, bytes)
#    if fast:
#        [(_, t1)] = ex.fast_encode_with_derivation(prev_token)
#        [(_, t2)] = ex.fast_encode_with_derivation(next_token)
#    else:
#        [_, [t1]] = ex.slow_encode_with_derivation(prev_token)
#        [_, [t2]] = ex.slow_encode_with_derivation(next_token)
#    return (_find_conflict(ex, t1, t2) is None)


class test_weird_tokens:
    def __init__(self, bpe):
        self.bpe = bpe

    def search(self, reps=100):
        for _ in iterview(range(reps)):
            token = np.random.choice(self.bpe._decode)
            if token.startswith(b'<|'): continue
            self(token)

    def __call__(self, token):
        #print(self.bpe.fast_encode_with_derivation(token))
        #print(self.bpe.slow_encode_with_derivation(token))
        #print(self.bpe.encode_as_token_ids(token))
        #print(self.bpe.encode_as_byte_chunks(token))
        #print(self.bpe.encode2(token))

        have = self.bpe.fast_encode_with_derivation(token)
        #print(self.bpe.slow_encode_with_derivation(token))
        #print(self.bpe.encode_as_token_ids(token))
        #print(self.bpe.encode_as_byte_chunks(token))
        want = self.bpe.encode2(token)
        assert len(have) == len(want), [token, have, want]
