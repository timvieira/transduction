import numpy as np
from arsenal.maths import sample_dict, sample


from functools import cached_property

# TODO: we should be able to merge this with LM.  The only challenge is in the
# "free monoid".  In the LM class, we use "nested tuples" (i.e., snoc lists),
# but here we use strings.
class CharLM:

    @cached_property
    def V(self):
        # Determine character alphabet from the LLM's vocabulary
        return {x for y in self.llm.V for x in y}

    def logp_next(self, context):
        raise NotImplementedError()

    def logprefix(self, context):
        if len(context) == 0:
            return 0.0
        else:
            context, y = context[:-1], context[-1]
            return self.logprefix(context) + self.logp_next(context)[y]

    def logp(self, context):
        return self.logprefix(context) + self.logp_next(context)[self.eos]

    def logp_next_seq(self, context, extension):
        """
        Compute `p(extension | context)` where `extension` is a sequence with |extension| > 1.
        """
        logP = 0
        for i in range(len(extension)):
            logp = self.logp_next(context + extension[:i])
            logP += logp[extension[i]]
        return logP

    def greedy(self, prompt, steps, verbose=False):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = prompt
        for _ in range(steps):
            p = self.logp_next(context)
            if verbose:
                print(repr(context), p.top(5))
            x = p.argmax()
            if x == self.eos:
                break
            context += x
        return context

    def sample(self, prompt, steps, verbose=False, draw=sample_dict):
        """
        Generate character-by-character starting from `prompt` using LLM with
        the approximate conditional distribution.
        """
        context = prompt
        for _ in range(steps):
            p = self.p_next(context)
            if verbose:
                print(repr(context), p.top(5))
            x = draw(p)
            if x == self.eos:
                break
            context += x
        return context

    def surprisals(self, context):
        s = []
        N = len(context)
#        for n in iterview(range(N+1), transient=True):
        for n in range(N):
            logp = self.logp_next(context[:n])
            s.append(-logp[context[n]])
        # And, finally, the surprisal of EOS
        logp = self.logp_next(context)
        s.append(-logp[self.eos])
        return s

    def plot_surprisals(self, context, **kwargs):
        from tokenization.util import plot_surprisals
        plot_surprisals(list(context) + [self.eos], self.surprisals(context), **kwargs)

    def guided_sample(self, guide, max_length, draw=sample_dict, verbosity=0):
        """
        Generate a simple guided sample (equivalent to SMC with one particle).
        """
        context = ''
        Z = 1
        for _ in range(max_length):
            u = guide.p_next(context)
            p = self.logp_next(context).map_values(np.exp)
            Q = p * u
            if not Q: break
            q = Q.normalize()
            y = draw(q)
            Z *= Q.sum()
            if verbosity > 0:
                print(context, y, [Z, Q.sum()], Q)
            if y == guide.eos: break
            context += y
        return (context, Z)

    # TODO: do computation in log-space to avoid underflow
    def smc(self, guide, n_particles, max_length, ess_threshold=0.5, verbosity=0):
        """
        Sequential Monte Carlo steering of LM using the `guide`.
        """
        beam = [('', 1, False)]*n_particles
        for _ in range(max_length):

            new_beam = []
            for context, weight, done in beam:

                if done:
                    new_beam.append((context, weight, True))
                    continue

                u = guide.p_next(context)
                p = self.logp_next(context).map_values(np.exp)
                Q = p * u

                # Particle hit a dead end, so its weight is zero
                if not Q:
                    new_beam.append((context, 0, True))
                    continue

                q = Q.normalize()
                y = sample_dict(q)

                if verbosity > 0:
                    print(context, y, [weight, Q.sum()], Q)

                new_beam.append((context + y, weight * Q.sum(), y == guide.eos))

            # resample if effective sample size dip below the threshold
            w = np.array([weight for _, weight, _ in new_beam])
            W = w.sum()
            if W**2 < ess_threshold * n_particles * (w*w).sum():
                W_avg = W/n_particles
                indices = sample(w, size=n_particles)
                new_beam = [(new_beam[i][0], W_avg, new_beam[i][2]) for i in indices]

            beam = new_beam

            if all(done for _, _, done in beam):
                break

        return beam

    # TODO: do computation in log-space to avoid underflow
    def swor_smc(self, guide, n_particles, max_length, verbosity=0):
        """
        Sequential without-replacement Monte Carlo steering of LM using the `guide`.
        """
        from swor.cps import ConditionalPoissonSampling

        beam = [('', 1, False)]
        for _ in range(max_length):

            candidates = []
            for (context, weight, done) in beam:

                if done:
                    candidates.append((context, weight, weight, True))
                    continue

                u = guide.p_next(context)
                p = self.logp_next(context).map_values(np.exp)
                Q = p * u

                for y, w in Q.items():
                    candidates.append((context + y, weight, weight * w, y == guide.eos))

            (contexts, old_weights, new_weight, dones) = zip(*candidates)
            cps = ConditionalPoissonSampling(np.array(new_weight), K=min(len(candidates), n_particles))
            subset = cps.sample()

            incl = cps.inclusion()

            beam = [
                (contexts[i], new_weight[i] / incl[i], dones[i])
                for i in subset
            ]

            if all(done for (_, _, done) in beam):
                break

        return beam

    def diverse_smc(self, guide, features, n_particles, max_length, verbosity=0):
        from dpp.kdpp import kDPP

        beam = [('', 1, False)]
        for _ in range(max_length):

            F = []; Ys = []; D = []; W = []
            for (context, weight, done) in beam:

                if done:
                    Ys.append(context)
                    W.append(weight)
                    F.append(features(context))
                    D.append(True)
                    continue

                u = guide.p_next(context)
                p = self.logp_next(context).map_values(np.exp)
                Q = p * u

                for x, px in Q.items():
                    Ys.append(context + x)
                    W.append(weight * px)
                    F.append(features(context + x))
                    D.append(x == self.eos)

            N = len(Ys)

            Q = np.sqrt(np.array(W))
            F = np.array(F)
            L = F @ F.T * np.outer(Q, Q)

            # TODO: warning L could have rank < k
            dpp = kDPP(L=L, k=min(N, n_particles))

            subset = dpp.sample()
            incl = dpp.inclusion()

            beam = [(Ys[i], W[i] / incl[i], D[i]) for i in subset]

        return beam
