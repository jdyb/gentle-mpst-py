#!/usr/bin/env python3

# The MIT License
# Copyright 2020 Jon Dybeck (jon <at> dybeck <dot> eu)
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, Tuple, Set, List, FrozenSet

class ExampleError(Exception):
    """This is raised if an example does not produce the expected results"""
    pass

# Section 3 "Synchronous Multiparty Session Calculus", Notation 01 (Base Sets)

class Label(object):
    def __init__(self, name: str):
        self.lname = name
    def __str__(self) -> str:
        return str(self.lname)
    def __repr__(self) -> str:
        return f'Label({self.lname})'
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Label):
            return self.lname == other.lname
        return NotImplemented
    def __hash__(self) -> int:
        return hash((Label, self.lname))

class Participant(object):
    """Session participant, from "Notation 01 (Base sets)" """
    def __init__(self, name: str):
        self.rname = name
    def __str__(self) -> str:
        return self.rname
    def __repr__(self) -> str:
        return f'Participant({self.rname})'
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Participant):
            return self.rname == other.rname
        return NotImplemented
    def __hash__(self) -> int:
        return hash((Participant, self.rname))

class Sort(object):
    """Sorts"""
    # FIXME From where are the sorts
    def __eq__(self, other: object) -> bool:
        # Prevent use of the default eq implementation.
        raise NotImplementedError()

class SNat(Sort):
    # FIXME Where is this defined
    def __str__(self) -> str:
        return 'nat'
    def __repr__(self) -> str:
        return 'SNat()'
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SNat)
    def __hash__(self) -> int:
        return hash(SNat)

class SInt(Sort):
    # FIXME Where is this defined
    def __repr__(self) -> str:
        return 'SInt()'
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SInt)
    def __hash__(self) -> int:
        return hash(SInt)

class SBool(Sort):
    # FIXME Where is this defined
    def __repr__(self) -> str:
        return 'SBool()'
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SBool)
    def __hash__(self) -> int:
        return hash(SBool)

# The subclasses of Local are from section "4.1 Types and Projectsions" and
# "Definition 3 (Local Session Types)"

class LocalT(object):
    """Local type."""

    def pt(self) -> Set[Participant]:
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        # Prevent use of the default eq implementation.
        raise NotImplementedError()

class LEnd(LocalT):
    """Local termination"""

    def pt(self) -> Set[Participant]:
        return set()

    def __repr__(self) -> str:
        return 'LEnd()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LEnd)

    def __hash__(self) -> int:
        return hash(LEnd)


class LExternalChoice(LocalT):
    """External Choice / Branching Type"""
    """The ampersand-thing and question-mark-thing"""

    def __init__(self, p: Participant, alternatives: Dict[Label, Tuple[Sort,LocalT]]):
        self.p, self.alternatives = p, alternatives

    def pt(self) -> Set[Participant]:
        pts = set((self.p,))
        for label in self.alternatives:
            sort, ltype = self.alternatives[label]
            pts.update(ltype.pt())
        return pts

    def __repr__(self) -> str:
        return f'LExternalChoice({repr(self.p)}, {repr(self.alternatives)})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LExternalChoice):
            return self.p == other.p and self.alternatives == other.alternatives
        return NotImplemented

    def __hash__(self) -> int:
        # Will not bother with recursing down the entire type now, this is
        # probably good enough to avoid most collisions.
        return hash((self.p, frozenset(self.alternatives.keys())))


class LInternalChoice(LocalT):
    """Internal Choice / Selection type"""
    """circle-cross and exclamationmarks"""

    def __init__(self, q: Participant, alternatives: Dict[Label, Tuple[Sort,LocalT]]):
        self.q, self.alternatives = q, alternatives

    def pt(self) -> Set[Participant]:
        pts = set((self.q,))
        for label in self.alternatives:
            sort, ltype = self.alternatives[label]
            pts.update(ltype.pt())
        return pts

    def __repr__(self) -> str:
        return f'LInternalChoice({repr(self.q)}, {repr(self.alternatives)})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LInternalChoice):
            return self.q == other.q and self.alternatives == other.alternatives
        return NotImplemented

    def __hash__(self) -> int:
        # Will not bother with recursing down the entire type now, this is
        # probably good enough to avoid most collisions.
        return hash((self.q, frozenset(self.alternatives.keys())))

class LVariable(LocalT):

    def __init__(self, name: str):
        self.ltvname = name

    def pt(self) -> Set[Participant]:
        return set()

    def __repr__(self) -> str:
        return f'LVariable({self.ltvname})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LVariable):
            return self.ltvname == other.ltvname
        return NotImplemented

    def __hash__(self) -> int:
        return hash((LVariable, self.ltvname))

class LRec(LocalT):

    def __init__(self, ltvariable: LVariable, local_type: LocalT):
        self.ltvariable, self.local_type = ltvariable, local_type

    def pt(self) -> Set[Participant]:
        return self.local_type.pt()

    def __repr__(self) -> str:
        return f'LRec({self.ltvariable}, {repr(self.local_type)})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LRec):
            return self.ltvariable == other.ltvariable and \
                    self.local_type == other.local_type
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.ltvariable, self.local_type))

# The subclasses of GlobalT are from section "4.1 Types and Projections"
# definition 2. 

class GlobalT(object):
    """Global type"""

    def pt(self) -> Set[Participant]:
        """Compute the set of participants of a global type.
        Returns the set of participants."""
        raise NotImplementedError()

    def project(self, r: Participant) -> LocalT:
        """Merging projection. See 'Definition 5'."""
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        # Prevent use of the default eq implementation.
        raise NotImplementedError()


class GEnd(GlobalT):
    """Global type signifying terminated protocol."""

    def pt(self):
        return set()

    def project(self, r: Participant) -> LocalT:
        """[PROJ-END]"""
        return LEnd()

    def __eq__(self, other):
        return isinstance(other, GEnd)

    def __hash__(self):
        return hash(GEnd)

class GTVar(GlobalT):
    """Global type variable"""

    def __init__(self, name: str):
        self.gtvname = name

    def pt(self):
        return set()

    def project(self, r):
        """[PROJ-VAR]"""
        return LVariable(self.gtvname)

class GRec(GlobalT):
    """Recursive global type"""

    def __init__(self, gtvariable: GTVar, global_type: GlobalT):
        self.gtvariable, self.global_type = gtvariable, global_type

    def pt(self):
        return self.global_type.pt()

    def project(self, r):
        """[PROJ-REC-1] [PROJ-REC-2]"""
        pts = self.global_type.pt()
        if r in pts:
            # [PROJ-REC-1]
            return LRec(self.gtvariable.project(r),
                    self.global_type.project(r))
        elif r not in pts:
            # [PROJ-REC-2]
            return LEnd()
        else:
            # Should never be reached.
            raise RuntimeError()

    def __eq__(self, other):
        if isinstance(other, GRec):
            return self.gtvariable == other.gtvariable and \
                    self.global_type == other.global_type
        return NotImplemented

    def __hash__(self):
        return hash((self.gtvariable, self.global_type))

class GCom(GlobalT):
    """Global type for message communication between two participants."""

    def __init__(self, source, destination, alternatives):
        """Alternatives is a dict, label keys."""
        self.source: Participant = source
        self.destination: Participant = destination
        self.alternatives: Dict[Label, Tuple[Sort, GlobalT]] = alternatives

    def pt(self) -> Set[Participant]:
        pts = set((self.source, self.destination))
        for label in self.alternatives:
            Si, Gi = self.alternatives[label]
            pts.add(Gi.pt())
        return pts

    def _proj_in(self, r: Participant) -> LExternalChoice:
        """ [PROJ-IN] """
        local_alternatives = {}
        for label in self.alternatives:
            sort, global_type = self.alternatives[label]
            local_alternatives[label] = (sort, global_type.project(r))
        return LExternalChoice(self.source, local_alternatives)

    def _proj_out(self, r: Participant) -> LInternalChoice:
        """ [PROJ-OUT] """
        local_alternatives = {}
        for label in self.alternatives:
            sort, global_type = self.alternatives[label]
            local_alternatives[label] = (sort, global_type.project(r))
        return LInternalChoice(self.destination, local_alternatives)

    def _proj_cont(self, r: Participant):
        """ [PROJ-CONT] """
        # FIXME Return type?
        # FIXME Check that alternatives has at least one element.
        # Project each of the continuation global types
        local_types: List[LocalT] = []
        for label in self.alternatives:
            sort, global_type = self.alternatives[label]
            local_types.append(global_type.project(r))
        # Now merge the local (session) types into a single local type. 
        merged_type = local_types.pop()
        for local_type in local_types:
            tmp = merge(merged_type, local_type)
            if tmp:
                merged_type = tmp
            else:
                return None
        return merged_type

    def project(self, r):
        """[PROJ-IN] [PROJ-OUT] [PROJ-CONT]"""
        if self.destination == r:
            return self._proj_in(r)
        elif self.source == r:
            return self._proj_out(r)
        elif self.source != r and self.destination != r:
            return self._proj_cont(r)
        else:
            # Should never be reached.
            raise RuntimeError()

    def __eq__(self, other):
        if isinstance(other, GCom):
            return self.source == other.source and \
                    self.destination == other.destination and \
                    self.alternatives == other.alternatives
        return NotImplemented

    def __hash__(self):
        return hash((self.source, self.destination,
            frozenset(self.alternatives.keys())))

def merge(T1: LocalT, T2: LocalT):
    """Merging operator."""
    """
    While this could be added to each LocalT subclass, that mostly just
    duplicates the [MRG-ID] case on everything but LExternalChoice, so we use
    a function instead.
    """
    # FIXME Return type annotation. May return None. 
    if T1 == T2:
        # [MRG-ID]
        return T1
    elif isinstance(T1, LExternalChoice) and isinstance(T2, LExternalChoice):
        # [MRG-BRA]
        #
        # FIXME Nicer way to write the condition?
        # Note that the definition of the merging operator is a little bit
        # unclear as to how to treat overlapping labels. The definition talks
        # about k being a member of the union of J and I, but it does not make
        # it clear how Tk is to be selected. Example 4 clarifies that when
        # I and J has overlapping labels then the continuation types Ti and Tj
        # must be the same. 
        #
        # Compute the label-sets, we will use these for iteration later.
        labels_t1: FrozenSet[Label] = frozenset(T1.alternatives.keys())
        labels_t2: FrozenSet[Label] = frozenset(T2.alternatives.keys())
        shared_labels = labels_t1.intersection(labels_t2)
        labels_only_t1 = labels_t1 - shared_labels
        labels_only_t2 = labels_t2 - shared_labels
        # Check that T1 and T2 have the same participant (p')
        if T1.p != T2.p:
            # The merge operator is undefined for this case.
            return None
        # Make empty T3 type, we will fill it in the following blocks of code.
        T3 = LExternalChoice(T1.p, {})
        # Collect type continuations exclusive to T1 into T3
        for label in labels_only_t1:
            T3.alternatives[label] = T1.alternatives[label]
        # Collect type continuations exclusive to T2 into T3
        for label in labels_only_t2:
            T3.alternatives[label] = T2.alternatives[label]
        # Check that the type continuations with labels shared between T1 and
        # T2 are the same. Then collect the type continuations into T3.
        # If the type continuations are not the same abort the merge.
        for label in shared_labels:
            t1c = T1.alternatives[label]
            t2c = T2.alternatives[label]
            if t1c == t2c:
                T3.alternatives[label] = t1c
            else:
                # The continuations do not match, so the merge is undefined.
                return None
        return T3
    else:
        return None

def example_4():
    # Participants and labels used in the example.
    p, q = Participant('p'), Participant('q')
    l, l3, l4, l5 = Label('l'), Label('l3'), Label('l4'), Label('l5')
    # First line in example.
    tst1 = LInternalChoice(q,{l:(SNat(),LEnd())})
    if merge(tst1, tst1) != tst1:
        raise ExampleError((example_4, 1))
    # Second line in example (undefined due to different participants).
    if merge(LInternalChoice(p, {l: (SNat(), LEnd())}),
            LInternalChoice(q,{l: (SNat(), LEnd())})) != None:
        raise ExampleError((example_4, 2))
    # Third line in example (undefined due to outputs with different labels)
    if merge(LInternalChoice(q,{l3:(SNat(), LEnd())}),
            LInternalChoice(q,{l4:(SNat(), LEnd())})) != None:
        raise ExampleError((example_4, 3))
    # Fourth line in example
    tmp4_l = LExternalChoice(q, {l3: (SInt(), LEnd()), l5: (SNat(), LEnd())})
    tmp4_r = LExternalChoice(q, {l4: (SInt(), LEnd()), l5: (SNat(), LEnd())})
    tmp4 = LExternalChoice(q, {l3: (SInt(), LEnd()), l4: (SInt(), LEnd()),
        l5: (SNat(), LEnd())})
    if merge(tmp4_l, tmp4_r) != tmp4:
        raise ExampleError((example_4, 4))
    # Fifth line in example
    tmp5_l = LExternalChoice(q, {l3: (SNat(), LEnd())})
    tmp5_r = LExternalChoice(q, {l3: (SNat(), LExternalChoice(q, {l3: (SNat(), LEnd())}))})
    if merge(tmp5_l, tmp5_r) != None:
        raise ExampleError((example_4, 5))

def project(G: GlobalT, r: Participant) -> LocalT:
    """Wrapper function for projection in definition 5"""
    return G.project(r)

class Expression(object):
    def __eq__(self, other):
        # Prevent use of the default eq implementation.
        raise NotImplementedError()

class Succ(Expression):
    def __init__(self, arg):
        self.arg = arg
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f'Succ({repr(self.arg)})'
    def eval(self, env):
        return 1 + eval_expr(self.arg, env)

def eval_expr(expr, env):
    if isinstance(expr, int):
        return expr
    else:
        return expr.eval(env)

class Variable(Expression):
    def __init__(self, name):
        self.vname = name
    def __str__(self):
        return self.vname
    def __repr__(self):
        return f'Variable({self.vname})'
    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.vname == other.vname
        return NotImplemented
    def __hash__(self):
        return hash((Variable, self.vname))
    def eval(self, env):
        return env[self]

class Process(object):
    def __init__(self):
        self.environment = {}
    def step(self, role, state):
        raise NotImplementedError()
    def comm(self, label, data):
        raise CannotCommunicate()

class Inaction(Process):
    def __init__(self):
        Process.__init__(self)
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f'Inaction({repr(self.environment)})'
    def step(self, role, state):
        # Nothing to step
        return None

class CannotCommunicate(Exception):
    pass # Intentionally empty exception.

class Send(Process):
    def __init__(self, destination, label, expr, continuation):
        Process.__init__(self)
        self.destination, self.label, self.expr, self.continuation = \
                destination, label, expr, continuation
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f'Send({self.destination}, {self.label}, {self.expr}, {self.continuation})'
    def step(self, role, state0):
        proc_dst = state0.participants[self.destination]
        data = eval_expr(self.expr, self.environment)
        try:
            proc_dst = proc_dst.comm(role, self.label, data)
            state1 = state0.replace(self.destination, proc_dst)
            # FIXME Eeeek. Do not mutate, plz.
            self.continuation.environment.update(self.environment)
            state2 = state1.replace(role, self.continuation)
            return state2
        except CannotCommunicate:
            # Cannot communicate with destination right now, so cannot step.
            return None

class Recv(Process):
    def __init__(self, source, label, variable, continuation):
        Process.__init__(self)
        self.source, self.label, self.variable, self.continuation = \
                source, label, variable, continuation
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f'Recv({self.source}, {self.label}, {self.variable}, {self.continuation})'
    def step(self, role, state):
        # Waiting for message, cannot step by ourselves.
        return None
    def comm(self, role, label, data):
        if self.source != role:
            # The other process is not the one we are waiting on. 
            raise CannotCommunicate()
        if self.label != label:
            # The label is not the one we are waiting on.
            raise CannotCommunicate()
        # Participant and Label matches, we will communicate (receive message).
        # FIXME Eeew. Do not mutate please.
        self.continuation.environment[self.variable] = data
        return self.continuation

class If(Process):
    def __init__(self, condition, positive, negative):
        Process.__init__(self)
        self.condition, self.positive, self.negative = \
                condition, positive, negative

class ExtChoice(Process):
    """An external choice."""
    def __init__(self, *alternatives):
        Process.__init__(self)
        """The alternatives must be a nonempty list of Recv processes."""
        self.alternatives = alternatives
    def __str__(self):
        return repr(self)
    def __repr__(self):
        return f'ExtChoice({repr(self.alternatives)})'
    def step(self, role, state):
        # We are waiting for another process, cannot step by ourselves.
        return None
    def comm(self, role, label, data):
        # Try each alternative and see if one of them will communicate.
        for proc in self.alternatives:
            try:
                return proc.comm(role, label, data)
            except CannotCommunicate:
                # No, try the other alternatives.
                continue
        # No alternative can communicate.
        raise CannotCommunicate()

class MState(object):
    def __init__(self, participants):
        Process.__init__(self)
        """Initialize state with dict of participants."""
        self.participants = dict(participants)
    def step(self):
        """Perform either one computation or communication step of the
        operational semantics.""" 
        # Search for a process that can step
        for role, proc in dict(self.participants).items():
            state = proc.step(role, self)
            if state:
                # Found a step, return new state
                return state
        # No process that can step was found.
        return None
    def replace(self, role, proc):
        participants1 = dict(self.participants)
        participants1[role] = proc
        return MState(participants1)
    def __repr__(self):
        strs = [f'\t{role}:\t{str(self.participants[role])}\n'
                for role in self.participants]
        return 'MState(\n{}\t)'.format(''.join(strs))
    def __eq__(self, other):
        # Prevent use of the default eq implementation.
        raise NotImplementedError()

def example_2():
    Bob, Alice, Carol = \
            Participant('Bob'), Participant('Alice'), Participant('Carol')
    l1, l2, l3, l4 = Label(1), Label(2), Label(3), Label(4)
    x = Variable('x')
    PAlice = Send(Bob, l1, 50, Recv(Carol, l3, x, Inaction()))
    PBob = ExtChoice(Recv(Alice, l1, x, Send(Carol, l2, 100, Inaction())),
            Recv(Alice, l4, x, Send(Carol, l2, 2, Inaction())))
    PCarol = Recv(Bob, l2, x, Send(Alice, l3, Succ(x), Inaction()))
    state = MState({Alice: PAlice, Bob: PBob, Carol: PCarol})
    while True:
        tmp = state.step()
        if tmp:
            state = tmp
        else:
            break
    if state.participants[Alice].environment[x] != 101:
        raise ExampleError(example_2)

def section_4_1_example_5():
    """Section 4.1 "Types and Projections", Example 5"""
    # Notes about this example
    #
    # Arrow in the publication becomes GCom class, int and nat are
    # SInt and SNat to avoid collision with Python reserved names. The GEnd is
    # omitted in the example, as stated at page 78
    # (Section 3 "Synchronous Multiparty Session Calculus"): 
    #   "We often omit 0 from the tail of processes"
    # Additionally this example uses infix operators for both internal and
    # external choice (the crossed circle and ampersands), these are
    # implemented as LInternalChoice and LExternalChoice constructions.
    #
    # Define the participants used in this example.
    q, p, r = Participant('q'), Participant('p'), Participant('r')
    # Define the labels used in this example.
    l1, l2, l3, l4, l5 = \
            Label('l1'), Label('l2'), Label('l3'), Label('l4'), Label('l5')
    # Define the global type
    G1 = GCom(q, r, {l3: (SInt(), GEnd()), l5: (SNat(), GEnd())})
    G2 = GCom(q, r, {l4: (SInt(), GEnd()), l5: (SNat(), GEnd())})
    G = GCom(p, q, {l1: (SNat(), G1), l2: (SBool(), G2)})
    # Local types of participants q, p and r
    Lq, Lp, Lr = G.project(q), G.project(p), G.project(r)
    # Expected local type of participant p
    Lp_ = LInternalChoice(q, {l1: (SNat(), LEnd()), l2: (SBool(), LEnd())})
    # Expected local type of participant q
    Lq_0 = LInternalChoice(r, {l3: (SInt(), LEnd()), l5: (SNat(), LEnd())})
    Lq_1 = LInternalChoice(r, {l5: (SNat(), LEnd()), l4: (SInt(), LEnd())})
    Lq_ = LExternalChoice(p, {l1: (SNat(), Lq_0), l2: (SBool(), Lq_1)})
    # FIXME BUG Lq has LExternalChoice as choices
    #   (Lq_0 and Lq_1 does not match).
    # Expected local type of participant r
    Lr_ = LExternalChoice(q, {l3: (SInt(), LEnd()), l4: (SInt(), LEnd()),
        l5: (SNat(), LEnd())})
    # Check that projected local types match the expected local types, to make
    # sure this example actually works.
    if Lq != Lq_ or Lp != Lp_ or Lr != Lr_:
        raise ExampleError((section_4_1_example_5, Lq, Lp, Lr))

def example_6():
    l1, l2, l3, l4 = Label('l1'), Label('l2'), Label('l3'), Label('l4')
    p, q, r = Participant('p'), Participant('q'), Participant('r')
    G1 = GCom(r, q, {l3: (SNat(), GEnd())})
    G2 = GCom(r, q, {l4: (SNat(), GEnd())})
    G = GCom(p, q, {l1: (SNat(), G1), l2: (SBool(), G2)})

    Gp = LInternalChoice(q, {l1: (SNat(), LEnd()), l2: (SBool(), LEnd())})
    Gq = LExternalChoice(p, {l1: (SNat(), LExternalChoice(r, {l3: (SNat(), LEnd())}))})

    if G.project(p) != Gp:
        raise ExampleError((example_6, 1))
    if G.project(q) != Gq:
        raise ExampleError((example_6, 2))
    if G.project(r) != None:
        raise ExampleError((example_6, 3))

example_2()
example_4()
section_4_1_example_5()
example_6()

