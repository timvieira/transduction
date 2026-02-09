"""
Context-dependent rewrite rule compilation (cdrewrite).

Implements the Mohri & Sproat (1996) algorithm for compiling rewrite rules of
the form ``phi -> psi / lambda __ rho`` into finite-state transducers.

Currently supports LTR obligatory mode only (the default in pynini's cdrewrite).

Reference implementation: pynini/src/cdrewrite.h (Google, Apache 2.0)
"""

from transduction.fst import FST, EPSILON
from transduction.fsa import FSA


# ---------------------------------------------------------------------------
# Marker symbols — sentinel objects distinct from any user alphabet symbol.
# ---------------------------------------------------------------------------

class _Marker:
    """Distinct sentinel for cdrewrite internal markers."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(('_Marker', self.name))

    def __eq__(self, other):
        return isinstance(other, _Marker) and self.name == other.name

    def __lt__(self, other):
        if isinstance(other, _Marker):
            return self.name < other.name
        return NotImplemented


BOS = _Marker('[BOS]')
EOS = _Marker('[EOS]')


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def cross(upper, lower):
    """Build a transducer mapping one string/FSA to another.

    Args:
        upper: string or FSA — the input side
        lower: string or FSA — the output side

    Returns:
        FST mapping upper's language to lower's language.
    """
    if isinstance(upper, str) and isinstance(lower, str):
        return _cross_strings(upper, lower)
    if isinstance(upper, str):
        upper = FSA.from_string(upper)
    if isinstance(lower, str):
        lower = FSA.from_string(lower)
    return _cross_fsa(upper, lower)


def _cross_strings(a, b):
    """Cross-product of two literal strings."""
    fst = FST()
    fst.add_start(0)
    n = max(len(a), len(b))
    for i in range(n):
        x = a[i] if i < len(a) else EPSILON
        y = b[i] if i < len(b) else EPSILON
        fst.add_arc(i, x, y, i + 1)
    fst.add_stop(n)
    return fst


def _cross_fsa(upper, lower):
    """Cross-product of two FSAs."""
    upper_paths = list(_extract_strings(upper, limit=100))
    lower_paths = list(_extract_strings(lower, limit=100))
    if len(upper_paths) == 1 and len(lower_paths) == 1:
        return _cross_strings(upper_paths[0], lower_paths[0])
    pairs = [(u, l) for u in upper_paths for l in lower_paths]
    return FST.from_pairs(pairs)


def _extract_strings(fsa, limit=100):
    """Extract strings from an FSA (up to limit)."""
    count = 0
    for s in fsa.language():
        yield s
        count += 1
        if count >= limit:
            break


def union_fst(*fsts):
    """Union of FSTs."""
    result = FST()
    for idx, fst in enumerate(fsts):
        for s in fst.start:
            result.add_start(('u', idx, s))
        for s in fst.stop:
            result.add_stop(('u', idx, s))
        for i in fst.states:
            for a, b, j in fst.arcs(i):
                result.add_arc(('u', idx, i), a, b, ('u', idx, j))
    return result


# ---------------------------------------------------------------------------
# Internal marker labels (allocated dynamically per Compile call)
# ---------------------------------------------------------------------------

# These are module-level sentinels used as FST arc labels.
# They must be distinct from any symbol in the user's alphabet.
_RBRACE  = _Marker('>')
_LBRACE1 = _Marker('<1')
_LBRACE2 = _Marker('<2')


# ---------------------------------------------------------------------------
# Helper: prepend sigma* to an FSA
# ---------------------------------------------------------------------------

def _prepend_sigma_star(fsa, sigma_fsa):
    """Return sigma* . fsa (concatenation)."""
    return sigma_fsa * fsa


# ---------------------------------------------------------------------------
# MakeMarker: modify a DFA (as an FSA) to insert/check markers
# ---------------------------------------------------------------------------

def _make_marker(dfa, marker_type, markers):
    """Transform a DFA into a marker transducer.

    The input DFA (an FSA from det+min) is converted into an FST that:
      - Has identity arcs (a:a) for all original arcs
      - Is modified at final/non-final states according to marker_type

    Faithfully follows MakeMarker in cdrewrite.h: for MARK type, final states
    have their arcs DELETED and moved to a new continuation state; only marker
    arcs remain on the original state.

    Args:
        dfa: FSA — determinized, minimized acceptor for sigma* . context
        marker_type: 'mark', 'check', or 'check_complement'
        markers: list of (input_label, output_label) pairs for marker arcs

    Returns:
        FST — the marker transducer
    """
    fst = FST()
    for s in dfa.start:
        fst.add_start(s)

    original_final = set(dfa.stop)

    if marker_type == 'mark':
        # MARK: Insert markers after each match.
        # Final states: delete arcs, add marker arcs to new continuation state
        #   which gets copies of the original arcs and finality.
        # Non-final states: become final, keep arcs.

        for s in dfa.states:
            if s not in original_final:
                # Non-final: copy arcs as identity, make final
                fst.add_stop(s)
                for a, j in dfa.arcs(s):
                    if a != EPSILON:
                        fst.add_arc(s, a, a, j)
            else:
                # Final: arcs DELETED from s, moved to new state
                new_s = ('marked', s)
                fst.add_stop(new_s)
                # New state gets copies of original arcs
                for a, j in dfa.arcs(s):
                    if a != EPSILON:
                        fst.add_arc(new_s, a, a, j)
                # Original state gets ONLY marker arcs (arcs were deleted)
                for (m_in, m_out) in markers:
                    fst.add_arc(s, m_in, m_out, new_s)

    elif marker_type == 'check':
        # CHECK: all states become final. At final states, add marker self-loops.
        for i, a, j in dfa.arcs():
            if a != EPSILON:
                fst.add_arc(i, a, a, j)
        for s in dfa.states:
            fst.add_stop(s)
            if s in original_final:
                for (m_in, m_out) in markers:
                    fst.add_arc(s, m_in, m_out, s)

    elif marker_type == 'check_complement':
        # CHECK_COMPLEMENT: all states become final.
        # At non-final states, add marker self-loops.
        for i, a, j in dfa.arcs():
            if a != EPSILON:
                fst.add_arc(i, a, a, j)
        for s in dfa.states:
            fst.add_stop(s)
            if s not in original_final:
                for (m_in, m_out) in markers:
                    fst.add_arc(s, m_in, m_out, s)

    return fst


# ---------------------------------------------------------------------------
# IgnoreMarkers: add marker self-loops at all states
# ---------------------------------------------------------------------------

def _ignore_markers(fst, markers):
    """Add self-loop arcs for each (input, output) marker pair at every state.

    This allows the FST to transparently pass through these markers at any position.
    """
    for s in list(fst.states):
        for (m_in, m_out) in markers:
            fst.add_arc(s, m_in, m_out, s)


# ---------------------------------------------------------------------------
# MakeFilter: the core primitive
# ---------------------------------------------------------------------------

def _make_filter(beta, sigma_fsa, marker_type, markers, reverse):
    """Build a context filter transducer.

    Implements MakeFilter from cdrewrite.h:
      1. If reverse: reverse beta and sigma
      2. Prepend sigma* to beta
      3. Epsilon-remove, determinize, minimize
      4. Apply MakeMarker
      5. If reverse: reverse the result FST

    Args:
        beta: FSA — the context pattern (lambda or rho)
        sigma_fsa: FSA — sigma* (closure of alphabet)
        marker_type: 'mark', 'check', or 'check_complement'
        markers: list of (input_label, output_label) pairs
        reverse: if True, reverse beta before processing, unreverse after

    Returns:
        FST — the filter transducer
    """
    # Work with the FSA
    pattern = beta

    if reverse:
        pattern = pattern.reverse()
        rev_sigma = sigma_fsa.reverse().epsremove()
        combined = _prepend_sigma_star(pattern, rev_sigma)
    else:
        combined = _prepend_sigma_star(pattern, sigma_fsa)

    # Epsilon-remove, determinize, minimize
    combined = combined.epsremove().det().min()

    # Apply marker transformation
    result_fst = _make_marker(combined, marker_type, markers)

    if reverse:
        result_fst = _reverse_fst(result_fst)

    return result_fst


def _reverse_fst(fst):
    """Reverse an FST: swap start/stop, reverse all arcs."""
    rev = FST()
    for s in fst.start:
        rev.add_stop(s)
    for s in fst.stop:
        rev.add_start(s)
    for i in fst.states:
        for a, b, j in fst.arcs(i):
            rev.add_arc(j, a, b, i)
    return rev


# ---------------------------------------------------------------------------
# MakeReplace: build the replace transducer
# ---------------------------------------------------------------------------

def _make_replace(tau, sigma_fsa):
    """Build the Replace transducer for LTR obligatory mode.

    Following cdrewrite.h MakeReplace for LEFT_TO_RIGHT + OBLIGATORY:
      - Entry: lbrace1:lbrace1 arc from new start to tau's start
      - Exit: rbrace:eps arc from tau's final states to new final state
      - Self-loops at all tau states: lbrace1:eps, lbrace2:eps, rbrace:eps
      - Self-loops at new initial state: lbrace2:lbrace2, rbrace:eps
      - Prepend (sigma ∪ {lbrace2:lbrace2, rbrace:eps})* to the whole thing
      - Apply Kleene star closure

    The result is: ((sigma ∪ lbrace2 ∪ rbrace)* lbrace1 tau rbrace)* (sigma ∪ ...)*
    """
    # Start with a copy of tau (as FST)
    fst = FST()
    for s in tau.start:
        fst.add_start(s)
    for s in tau.stop:
        fst.add_stop(s)
    for i in tau.states:
        for a, b, j in tau.arcs(i):
            fst.add_arc(i, a, b, j)

    # Add self-loops for markers at ALL states of tau
    all_loops = [(_LBRACE1, EPSILON), (_LBRACE2, EPSILON), (_RBRACE, EPSILON)]
    _ignore_markers(fst, all_loops)

    # Create new start and final states
    new_start = ('replace', 'start')
    new_final = ('replace', 'final')

    # Entry arc: lbrace1:lbrace1 from new_start to old start states
    for s in list(fst.start):
        fst.add_arc(new_start, _LBRACE1, _LBRACE1, s)

    # Exit arcs: rbrace:eps from old final states to new_final
    for s in list(fst.stop):
        fst.add_arc(s, _RBRACE, EPSILON, new_final)
        fst.stop.discard(s)

    fst.add_stop(new_final)
    fst.add_stop(new_start)  # new_start is also final (for empty match / pass-through)
    fst.start = {new_start}

    # Initial state loops: lbrace2:lbrace2, rbrace:eps
    initial_loops = [(_LBRACE2, _LBRACE2), (_RBRACE, EPSILON)]

    # Prepend (sigma ∪ initial_loops)* to fst
    # Build the sigma FSA extended with the initial markers
    sigma_ext_fsa = FSA()
    for s in sigma_fsa.start:
        sigma_ext_fsa.add_start(s)
    for s in sigma_fsa.stop:
        sigma_ext_fsa.add_stop(s)
    for i, a, j in sigma_fsa.arcs():
        sigma_ext_fsa.add_arc(i, a, j)
    # Add marker arcs to sigma_ext_fsa
    for s in list(sigma_ext_fsa.stop):
        for s2 in list(sigma_ext_fsa.start):
            for (m_in, _) in initial_loops:
                sigma_ext_fsa.add_arc(s, m_in, s2)

    # Instead of literal prepend + closure (which would be complex),
    # we do the equivalent directly:
    # The replace transducer should be: (sigma_ext*)  (lbrace1 tau rbrace)?  repeated
    #
    # Which is: any number of sigma/lbrace2/rbrace symbols, optionally interleaved
    # with (lbrace1 tau rbrace) sequences.
    #
    # We build this as: add sigma identity loops at new_start, plus the initial_loops.

    for s in list(sigma_fsa.states):
        for a, j in sigma_fsa.arcs(s):
            if a != EPSILON:
                # Map sigma_fsa state 0 to new_start
                src = new_start
                dst = new_start
                fst.add_arc(src, a, a, dst)

    # Add initial_loops at new_start
    for (m_in, m_out) in initial_loops:
        fst.add_arc(new_start, m_in, m_out, new_start)

    # After completing a rewrite (at new_final), loop back to allow more rewrites
    # new_final should act like new_start for continuation
    for a in sigma_fsa.syms - {EPSILON}:
        fst.add_arc(new_final, a, a, new_final)
    for (m_in, m_out) in initial_loops:
        fst.add_arc(new_final, m_in, m_out, new_final)
    # Allow starting a new rewrite from new_final
    for s in tau.start:
        fst.add_arc(new_final, _LBRACE1, _LBRACE1, s)

    return fst


# ---------------------------------------------------------------------------
# Top-level cdrewrite
# ---------------------------------------------------------------------------

def cdrewrite(tau, lambda_=None, rho=None, sigma_star=None,
              direction='ltr', mode='obl'):
    """Compile a context-dependent rewrite rule into an FST.

    Implements: phi -> psi / lambda __ rho

    Args:
        tau: FST — the transduction (phi -> psi).  Build with ``cross()``.
        lambda_: FSA or None — left context (None = no left context)
        rho: FSA or None — right context (None = no right context)
        sigma_star: FSA — closure over the full alphabet.
                    Build with ``FSA.universal(alphabet)``.
        direction: ``'ltr'`` — left-to-right (only supported mode)
        mode: ``'obl'`` — obligatory (only supported mode)

    Returns:
        FST implementing the rewrite rule.
    """
    if direction != 'ltr' or mode != 'obl':
        raise NotImplementedError(
            f'Only ltr/obl mode is supported, got {direction}/{mode}'
        )
    if sigma_star is None:
        raise ValueError('sigma_star is required')

    # Default empty contexts
    if lambda_ is None:
        lambda_ = _epsilon_fsa()
    if rho is None:
        rho = _epsilon_fsa()

    # Check for BOS/EOS in contexts
    has_bos = BOS in lambda_.syms or BOS in tau.A
    has_eos = EOS in rho.syms or EOS in tau.A

    sigma = sigma_star.syms - {EPSILON}

    if has_bos or has_eos:
        return _compile_with_boundaries(tau, lambda_, rho, sigma_star, sigma,
                                        has_bos, has_eos)

    return _compile_ltr_obl(tau, lambda_, rho, sigma_star)


def _epsilon_fsa():
    """FSA accepting only the empty string."""
    m = FSA()
    m.add_start(0)
    m.add_stop(0)
    return m


def _compile_with_boundaries(tau, lambda_, rho, sigma_star, sigma,
                              has_bos, has_eos):
    """Handle BOS/EOS by wrapping with boundary inserter/deleter."""
    # Extend sigma_star with boundary markers
    ext_sigma_star = FSA()
    for s in sigma_star.start:
        ext_sigma_star.add_start(s)
    for s in sigma_star.stop:
        ext_sigma_star.add_stop(s)
    for i, a, j in sigma_star.arcs():
        ext_sigma_star.add_arc(i, a, j)
    if has_bos:
        for s in ext_sigma_star.stop:
            for s2 in ext_sigma_star.start:
                ext_sigma_star.add_arc(s, BOS, s2)
    if has_eos:
        for s in ext_sigma_star.stop:
            for s2 in ext_sigma_star.start:
                ext_sigma_star.add_arc(s, EOS, s2)

    # Build core with extended alphabet
    core = _compile_ltr_obl(tau, lambda_, rho, ext_sigma_star)

    # Build boundary inserter: eps:BOS sigma* eps:EOS
    ins = FST()
    state = 0
    ins.add_start(state)
    if has_bos:
        ins.add_arc(state, EPSILON, BOS, state + 1)
        state += 1
    body = state
    for a in sigma:
        ins.add_arc(body, a, a, body)
    if has_eos:
        ins.add_arc(body, EPSILON, EOS, state + 1)
        state += 1
        ins.add_stop(state)
    else:
        ins.add_stop(body)

    # Build boundary deleter: BOS:eps sigma* EOS:eps
    delete = FST()
    state = 0
    delete.add_start(state)
    if has_bos:
        delete.add_arc(state, BOS, EPSILON, state + 1)
        state += 1
    body = state
    for a in sigma:
        delete.add_arc(body, a, a, body)
    if has_eos:
        delete.add_arc(body, EOS, EPSILON, state + 1)
        state += 1
        delete.add_stop(state)
    else:
        delete.add_stop(body)

    result = ins @ core @ delete
    return result.trim()


def _compile_ltr_obl(tau, lambda_, rho, sigma_star):
    """Compile LTR obligatory cdrewrite using the 5-transducer pipeline.

    Pipeline: (((r @ f) @ replace) @ l1) @ l2

    Where:
        r:       right-context filter (MARK, reversed) — inserts RBRACE
        f:       phi-pattern filter (MARK, reversed) — inserts LBRACE1/LBRACE2
        replace: performs phi->psi, consuming markers
        l1:      left-context CHECK — consumes LBRACE1 at lambda-match positions
        l2:      left-context CHECK_COMPLEMENT — consumes LBRACE2 at non-match positions
    """
    # Extract phi from tau
    phi = tau.project(0).trim()
    # Convert phi FSA to have RBRACE self-loops at all states and RBRACE appended
    phi_fsa = _fsa_from_fst_input(tau)

    sigma_fsa = sigma_star  # This is already sigma*

    # The output of tau (psi) may contain symbols not in sigma.
    # l1 and l2 filters need to pass through these symbols too.
    psi_extra = (tau.B - {EPSILON}) - (sigma_fsa.syms - {EPSILON})
    if psi_extra:
        sigma_fsa_ext = _extend_sigma_star(sigma_fsa, list(psi_extra))
    else:
        sigma_fsa_ext = sigma_fsa

    # sigma_rbrace = sigma ∪ {RBRACE} (as sigma* FSA)
    sigma_rbrace_fsa = _extend_sigma_star(sigma_fsa, [_RBRACE])

    # --- r filter ---
    # MakeFilter(rho, sigma, MARK, {(eps, rbrace)}, reverse=true)
    r = _make_filter(rho, sigma_fsa, 'mark', [(EPSILON, _RBRACE)], reverse=True)

    # --- f filter ---
    # phi_rbrace: phi with RBRACE self-loops, followed by RBRACE
    phi_rbrace = _build_phi_rbrace(phi_fsa)
    # MakeFilter(phi_rbrace, sigma_rbrace, MARK, {(eps, lbrace1), (eps, lbrace2)}, reverse=true)
    f = _make_filter(phi_rbrace, sigma_rbrace_fsa, 'mark',
                     [(EPSILON, _LBRACE1), (EPSILON, _LBRACE2)], reverse=True)

    # --- replace ---
    # Use extended sigma so replace can pass through psi-only symbols
    replace = _make_replace(tau, sigma_fsa_ext)

    # --- l1 filter ---
    # Use extended sigma so l1 can pass through psi-only symbols
    l1 = _make_filter(lambda_, sigma_fsa_ext, 'check',
                      [(_LBRACE1, EPSILON)], reverse=False)
    # IgnoreMarkers: add lbrace2:lbrace2 self-loops
    _ignore_markers(l1, [(_LBRACE2, _LBRACE2)])

    # --- l2 filter ---
    l2 = _make_filter(lambda_, sigma_fsa_ext, 'check_complement',
                      [(_LBRACE2, EPSILON)], reverse=False)

    # Compose: (((r @ f) @ replace) @ l1) @ l2
    result = r @ f
    result = result @ replace
    result = result @ l1
    result = result @ l2

    return result.trim()


# ---------------------------------------------------------------------------
# Additional helpers
# ---------------------------------------------------------------------------

def _fsa_from_fst_input(tau):
    """Extract the input projection of an FST as an FSA."""
    fsa = FSA()
    for s in tau.start:
        fsa.add_start(s)
    for s in tau.stop:
        fsa.add_stop(s)
    for i in tau.states:
        for a, b, j in tau.arcs(i):
            fsa.add_arc(i, a, j)
    return fsa


def _build_phi_rbrace(phi_fsa):
    """Build phi with RBRACE self-loops at all states, followed by RBRACE.

    This is: IgnoreMarkers(phi, {(rbrace, rbrace)}); AppendMarkers(phi, {(rbrace, rbrace)})
    """
    # Copy phi
    result = FSA()
    for s in phi_fsa.start:
        result.add_start(s)
    for i, a, j in phi_fsa.arcs():
        result.add_arc(i, a, j)
    # Add RBRACE self-loops at all states
    for s in phi_fsa.states:
        result.add_arc(s, _RBRACE, s)
    # Append RBRACE: create new final state, connect old finals via RBRACE
    new_final = ('phi_rbrace', 'final')
    for s in phi_fsa.stop:
        result.add_arc(s, _RBRACE, new_final)
    result.add_stop(new_final)
    return result


def _extend_sigma_star(sigma_fsa, extra_symbols):
    """Extend sigma* FSA with additional symbols."""
    result = FSA()
    for s in sigma_fsa.start:
        result.add_start(s)
    for s in sigma_fsa.stop:
        result.add_stop(s)
    for i, a, j in sigma_fsa.arcs():
        result.add_arc(i, a, j)
    # Add extra symbol arcs: at final states, loop through start
    for s in list(sigma_fsa.stop):
        for s2 in list(sigma_fsa.start):
            for sym in extra_symbols:
                result.add_arc(s, sym, s2)
    return result
