import math
import random
from dataclasses import dataclass
from typing import FrozenSet, Tuple, Any
from transduction import FST, EPSILON, Precover

# TODO: put in constructor
from transduction.smc.aa import score_sequence


def logsumexp(log_probs):
    if not log_probs:
        return float('-inf')
    
    max_score = max(log_probs)
    if max_score == float('-inf'):
        return float('-inf')
        
    return max_score + math.log(sum(math.exp(x - max_score) for x in log_probs))



def sample_from_log_probs(tokens, log_probs, log_Z):
    r = math.log(random.random()) + log_Z
    
    cumulative = float('-inf')
    for t, lp in zip(tokens, log_probs):
        cumulative = logsumexp([cumulative, lp])
        if r < cumulative:
            return t
    
    return tokens[-1] # Fallback for floating point errors


def compute_epsilon_closure(frontier, fst, target_y):
    closure = set(frontier)
    worklist = list(frontier)
    
    while worklist:
        s, ys = worklist.pop()
        for output_sym, dests in fst.delta[s][EPSILON].items():
            next_ys = extend_output(ys, output_sym)
            if is_compatible(next_ys, target_y):
                for next_state in dests:
                    new_config = (next_state, next_ys)
                    if new_config not in closure:
                        closure.add(new_config)
                        worklist.append(new_config)
                            
    return closure


# We track (FST_State_Index, Output_String_So_Far)
StateTuple = Tuple[int, str]


@dataclass
class Particle:
    x: str
    states: FrozenSet[StateTuple] 
    weight: float
    is_universal: bool = False
    is_complete: bool = False

    @property
    def __str__(self):
        return "".join(self.x)

    @classmethod
    def initial(cls, start_states, fst, y):
        raw_frontier = set((s, "") for s in start_states)
        
        initial_frontier = frozenset(
            compute_epsilon_closure(raw_frontier, fst, y)
        )
        
        return cls(
            x="",
            states=initial_frontier,
            weight=0.0,
            is_universal=False,
            is_complete=False
        )

    def __repr__(self):
        status = "Univ" if self.is_universal else ("Done" if self.is_complete else "Active")
        return f"π(x={self.x}, #states={len(self.states)}, w={self.weight:.2f}, {status})"


class SMC:
    def __init__(self, fst, algo_class, tgt_str, lm, resample_threshold=0.7, num_particles=100, max_steps=50):
        self.fst = fst 
        self.algo = algo_class(fst, extend=lambda x, y: x + y)
        self.particles = [Particle.initial(fst.I, fst, tgt_str)]
        self.lm = lm
        self.tgt_str = tgt_str
        self.num_particles = num_particles
        self.max_steps = max_steps
        self.resample_threshold = resample_threshold

    def get_valid_proposal_tokens(self, p):
        # since epsilon outputs, need to scan ahead
        
        cur_out = next(iter(self.fst.transduce(p.x)))
        
        # scan until emit
        cands = self.algo.candidates(p.x, self.tgt_str)
        new_cands = []
        work_list = list(cands)
        while work_list:
            cand = work_list.pop()
            candy = next(iter(self.fst.transduce(cand)))
            if len(candy) > len(cur_out):
                new_cands.append(cand)
                continue
            work_list += list(self.algo.candidates(cand, self.tgt_str))
        return [s[len(p.x)] for s in new_cands]
        
    def get_proposal(self, dist, p):
        """
        q(x) propto p(x) * 1(x in A(S)).
        """
        valid_dist = {}
        valid_cands = self.get_valid_proposal_tokens(p)
        for token, log_p in dist.items():
            if log_p == float('-inf'): 
                continue
            
            if token not in valid_cands:
                continue
                
            valid_dist[token] = log_p
                
        if not valid_dist:
            return {}
            
        log_Z = logsumexp(valid_dist.values())
        proposal_dist = {
            t: lp - log_Z 
            for t, lp in valid_dist.items()
        }
        
        return proposal_dist

    def sample(self, dist):
        r = math.log(random.random())
        cumulative = float('-inf')
        for t, lp in dist.items():
            cumulative = logsumexp([cumulative, lp])
            if r < cumulative:
                return t
        return next(reversed(dist.keys()))
        
    def smc_step(self):
        new_particles = []

        for p in self.particles:
            if p.is_universal and p.is_complete:
                # since universal and extends,
                # we have already caught all mass
                new_particles.append(p)
                continue

            seq_str = str(p.x)
            raw_lm_dist = self.lm(seq_str)

            proposal_dist = self.get_proposal(raw_lm_dist, p)
            if not proposal_dist:
                continue
            elif not proposal_dist and p.is_complete:
                # add the particle back
                new_particles.append(p)
                continue

            # we know its a candidate
            sampled_token = self.sample(proposal_dist)
            new_weight = p.weight + proposal_dist[sampled_token]
            new_x = self.algo.extend(p.x, sampled_token)            
            # This is more than just a path since it means
            # we check many paths at once.
            # todo: make optional
            is_univ = self.algo.continuity(new_x, self.tgt_str)

            states = {s for s, _ in p.states}
            is_final = not self.fst.F.isdisjoint(states)
            new_y = next(iter(self.fst.transduce(new_x)))
            is_complete = new_y.startswith(self.tgt_str) and is_final

            new_states = self.algo.next_frontier(
                p.states, sampled_token)
            
            new_particles.append(Particle(
                x=new_x,
                states=new_states,
                weight=new_weight,
                is_universal=is_univ,
                is_complete=is_complete
            ))

        return new_particles
        
    def resample_particles(self, resample_threshold):
        if not self.particles:
            return self.particles

        log_weights = [p.weight for p in self.particles]
        log_sum_w = logsumexp(log_weights)
        log_sum_sq_w = logsumexp([2 * w for w in log_weights])
        
        log_ess = 2 * log_sum_w - log_sum_sq_w
        ess = math.exp(log_ess)
        
        N = len(self.particles)
        
        if ess >= resample_threshold * N:
            return self.particles
            
        # sample new to duplicate
        # todo: consider not uniform
        # todo: consider old particles for backtracking
        max_w = max(log_weights)
        linear_weights = [math.exp(w - max_w) for w in log_weights]
        
        selected_indices = random.choices(
            population=range(N),
            weights=linear_weights,
            k=self.num_particles - N
        )
        
        new_particles = []
        new_log_weight = log_sum_w - math.log(N)
        
        for idx in selected_indices:
            parent = self.particles[idx]
            new_particles.append(Particle(
                x=parent.x,
                states=parent.states,
                weight=new_log_weight,
                is_universal=parent.is_universal,
                is_complete=parent.is_complete
            ))
            
        return new_particles
    
    def __call__(self, resample_threshold=0.5):
        
        self.resample_threshold = resample_threshold

        for t in range(self.max_steps):
            if not self.particles:
                print(f"All particles died at step {t}")
                break
                
            if all(p.is_complete for p in self.particles):
                break
                
            self.particles = self.smc_step()
            # Resample
            self.particles = self.resample_particles(resample_threshold)

        return self.particles

    def get_probs(self):
        probs = 0.0
        samp_cover = self()
        for p in samp_cover:
            if p.is_universal:
                probs += score_sequence(p.x)
            # todo, add eos and handle remainder
        return probs

    @classmethod
    def get_dist(cls, fst, algo, tgt_str, lm, nump):
        dist = {}

        for symbol in fst.B:
            # todo, better extend
            ssmc = cls(fst, algo, tgt_str + symbol, lm, num_particles=nump)
            dist[symbol] = ssmc.get_probs()

        Z = logsumexp(dist.values())
        return {k: v - Z for k, v in dist.items()}