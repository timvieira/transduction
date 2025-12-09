import math

from transduction import EPSILON, FST

CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

AA = set(CODON_TABLE.values())

def create_dna_translator():
    m = FST()
    
    start_node = 0
    m.add_I(start_node)
    m.add_F(start_node)
    
    for codon, amino_acid in CODON_TABLE.items():
        n1, n2, n3 = codon[0], codon[1], codon[2]
        state_seen_1 = (n1,)
        state_seen_2 = (n1, n2)
        
        m.add_arc(start_node, n1, EPSILON, state_seen_1)
        m.add_arc(state_seen_1, n2, EPSILON, state_seen_2)
        m.add_arc(state_seen_2, n3, amino_acid, start_node)
        
        # to use quotient!
        m.add_F(state_seen_1)
        m.add_F(state_seen_2)

    return m


BIGRAM_PROBS = {
    'A': {'A': 0.2, 'C': 0.2, 'G': 0.3, 'T': 0.3},
    'C': {'A': 0.1, 'C': 0.4, 'G': 0.1, 'T': 0.4},
    'G': {'A': 0.3, 'C': 0.3, 'G': 0.2, 'T': 0.2},
    'T': {'A': 0.2, 'C': 0.2, 'G': 0.2, 'T': 0.4}
}
START_PROBS = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}



def score_sequence(sequence, log_space=True):
    if not sequence:
        return 0.0
        
    first = sequence[0]
    if first not in START_PROBS:
        raise ValueError(f"Invalid: {first}")
        
    current_score = START_PROBS[first]
    
    if log_space:
        current_score = math.log(current_score)

    for i in range(1, len(sequence)):
        prev = sequence[i-1]
        curr = sequence[i]
        
        p_trans = BIGRAM_PROBS[prev][curr]
        
        if log_space:
            current_score += math.log(p_trans)
        else:
            current_score *= p_trans
            
    return current_score


def get_source_lm_probs(seq):
    dist = {}
    
    if not seq:
        log_Z = math.log(sum(START_PROBS.values()))
        return {k: math.log(p) - log_Z for k, p in START_PROBS.items()}

    prev_char = seq[-1]
    if prev_char not in BIGRAM_PROBS:
        return {k: float('-inf') for k in BIGRAM_PROBS.keys()}
        
    row = BIGRAM_PROBS[prev_char]
    
    # in case not normalized..
    # this should not be necessary if model correct
    row_sum = sum(row.values())
    log_Z = math.log(row_sum)
    
    for token in BIGRAM_PROBS.keys():
        if token in row:
            dist[token] = math.log(row[token]) - log_Z
        else:
            dist[token] = float('-inf')
            
    return dist
