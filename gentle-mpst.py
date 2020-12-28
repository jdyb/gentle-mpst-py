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

from typing import Dict, Tuple, Set, List, FrozenSet, Optional, Any

class ExampleError(Exception):
    """This is raised if an example does not produce the expected results"""
    pass

# Section 3 "Synchronous Multiparty Session Calculus", Notation 01 (Base Sets)

class Label(object):
    def __init__(self, name: int):
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
    def is_subsort(self, other: 'Sort') -> bool:
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
    def is_subsort(self, other: Sort) -> bool:
        # The subsort relation is reflexive, so SNat <=: SNat.
        # Also, per Definition 6, SNat <=: SInt
        return isinstance(other, SNat) or isinstance(other, SInt)

class SInt(Sort):
    # FIXME Where is this defined
    def __repr__(self) -> str:
        return 'SInt()'
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SInt)
    def __hash__(self) -> int:
        return hash(SInt)
    def is_subsort(self, other: Sort) -> bool:
        # SInt is not a subsort of any other sort.
        return isinstance(other, SInt)

class SBool(Sort):
    # FIXME Where is this defined
    def __repr__(self) -> str:
        return 'SBool()'
    def __eq__(self, other: object) -> bool:
        return isinstance(other, SBool)
    def __hash__(self) -> int:
        return hash(SBool)
    def is_subsort(self, other: Sort) -> bool:
        # SBool is not a subsort of any other sort.
        return isinstance(other, SBool)

# The subclasses of Local are from section "4.1 Types and Projections" and
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

    def project(self, r: Participant) -> Optional[LocalT]:
        """Merging projection. See 'Definition 5'."""
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        # Prevent use of the default eq implementation.
        raise NotImplementedError()


class GEnd(GlobalT):
    """Global type signifying terminated protocol."""

    def pt(self) -> Set[Participant]:
        return set()

    def project(self, r: Participant) -> LocalT:
        """[PROJ-END]"""
        return LEnd()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GEnd)

    def __hash__(self) -> int:
        return hash(GEnd)

class GTVar(GlobalT):
    """Global type variable"""

    def __init__(self, name: str):
        self.gtvname = name

    def pt(self) -> Set[Participant]:
        return set()

    def project(self, r: Participant) -> LVariable:
        """[PROJ-VAR]"""
        return LVariable(self.gtvname)

class GRec(GlobalT):
    """Recursive global type"""

    def __init__(self, gtvariable: GTVar, global_type: GlobalT):
        self.gtvariable, self.global_type = gtvariable, global_type

    def pt(self) -> Set[Participant]:
        return self.global_type.pt()

    def project(self, r: Participant) -> Optional[LocalT]:
        """[PROJ-REC-1] [PROJ-REC-2]"""
        pts = self.global_type.pt()
        if r in pts:
            # [PROJ-REC-1]
            tmp = self.global_type.project(r)
            if tmp:
                return LRec(self.gtvariable.project(r), tmp)
            return None
        elif r not in pts:
            # [PROJ-REC-2]
            return LEnd()
        else:
            # Should never be reached.
            raise RuntimeError()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GRec):
            return self.gtvariable == other.gtvariable and \
                    self.global_type == other.global_type
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.gtvariable, self.global_type))

class GCom(GlobalT):
    """Global type for message communication between two participants."""

    def __init__(self, source: Participant, destination: Participant,
            alternatives: Dict[Label, Tuple[Sort, GlobalT]]):
        """Alternatives is a dict, label keys."""
        self.source, self.destination, self.alternatives = \
                source, destination, alternatives

    def pt(self) -> Set[Participant]:
        pts = set((self.source, self.destination))
        for label in self.alternatives:
            Si, Gi = self.alternatives[label]
            pts.update(Gi.pt())
        return pts

    def _proj_in(self, r: Participant) -> Optional[LExternalChoice]:
        """ [PROJ-IN] """
        local_alternatives = {}
        for label in self.alternatives:
            sort, global_type = self.alternatives[label]
            tmp = global_type.project(r)
            if tmp:
                local_alternatives[label] = (sort, tmp)
            else:
                # It was not possible to project global_type onto r.
                return None
        return LExternalChoice(self.source, local_alternatives)

    def _proj_out(self, r: Participant) -> Optional[LInternalChoice]:
        """ [PROJ-OUT] """
        local_alternatives = {}
        for label in self.alternatives:
            sort, global_type = self.alternatives[label]
            tmp = global_type.project(r)
            if tmp:
                local_alternatives[label] = (sort, tmp)
            else:
                # It was not possible to project global_type onto r.
                return None
        return LInternalChoice(self.destination, local_alternatives)

    def _proj_cont(self, r: Participant) -> Optional[LocalT]:
        """ [PROJ-CONT] """
        # FIXME Return type?
        # FIXME Check that alternatives has at least one element.
        # Project each of the continuation global types
        local_types: List[LocalT] = []
        for label in self.alternatives:
            sort, global_type = self.alternatives[label]
            tmp = global_type.project(r)
            if tmp:
                local_types.append(tmp)
            else:
                # It was not possible to project global_type onto r.
                return None
        # Now merge the local (session) types into a single local type.
        merged_type = local_types.pop()
        for local_type in local_types:
            tmp = merge(merged_type, local_type)
            if tmp:
                merged_type = tmp
            else:
                return None
        return merged_type

    def project(self, r: Participant) -> Optional[LocalT]:
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

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GCom):
            return self.source == other.source and \
                    self.destination == other.destination and \
                    self.alternatives == other.alternatives
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.source, self.destination,
            frozenset(self.alternatives.keys())))

def merge(T1: LocalT, T2: LocalT) -> Optional[LocalT]:
    """Merging operator."""
    """
    While this could be added to each LocalT subclass, that mostly just
    duplicates the [MRG-ID] case on everything but LExternalChoice, so we use
    a function instead.
    """
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

def example_4() -> None:
    # Participants and labels used in the example.
    p, q = Participant('p'), Participant('q')
    l, l3, l4, l5 = Label(0), Label(3), Label(4), Label(5)
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

class TypingEnvironment(object):
    """Maps expression variables to sorts and process variables to session types.
    From Section 4.3 "Type System". Used by typechecking and inference."""
    def __init__(self) -> None:
        # Expression environment
        self.e_env: Dict[Variable, Sort] = {}
        # (Local) Process environment
        self.p_env: Dict[LVariable, LocalT] = {}
        # FIXME Add multiparty session to global type
    def lookup_variable(self, var: 'Variable') -> Sort:
        return self.e_env[var]
    def lookup_lvariable(self, lvar: LVariable) -> LocalT:
        return self.p_env[lvar]
    def bind_variable(self, var: 'Variable' , srt: Sort) -> 'TypingEnvironment':
        te = TypingEnvironment()
        te.e_env = self.e_env.copy()
        te.e_env[var] = srt
        te.p_env = self.p_env
        return te
    def bind_lvariable(self, lvar: LVariable, ltype: LocalT) -> 'TypingEnvironment':
        te = TypingEnvironment()
        te.p_env = self.p_env.copy()
        te.p_env[lvar] = ltype
        te.e_env = self.e_env
        return te
    def __str__(self) -> str:
        return repr(self)
    def __repr__(self) -> str:
        return f'TypingEnvironment({self.e_env},{self.p_env})'

class Expression(object):
    def __eq__(self, other: object) -> bool:
        # Prevent use of the default eq implementation.
        raise NotImplementedError()
    def eval(self, env: Dict['Variable', Any]) -> Any:
        # FIXME Replace Any with a value class.
        raise NotImplementedError()
    def typecheck(self, the_type: Sort, tenv: TypingEnvironment) -> bool:
        raise NotImplementedError()

class Variable(Expression):
    def __init__(self, name: str):
        self.vname = name
    def __str__(self) -> str:
        return self.vname
    def __repr__(self) -> str:
        return f'Variable({self.vname})'
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Variable):
            return self.vname == other.vname
        return NotImplemented
    def __hash__(self) -> int:
        return hash((Variable, self.vname))
    def eval(self, env: Dict['Variable', Any]) -> Any:
        return env[self]
    def typecheck(self, the_type: Sort, tenv: TypingEnvironment) -> bool:
        return tenv.e_env[self] == the_type

class Succ(Expression):
    def __init__(self, arg: Expression):
        self.arg = arg
    def __str__(self) -> str:
        return repr(self)
    def __repr__(self) -> str:
        return f'Succ({repr(self.arg)})'
    def eval(self, env: Dict[Variable, Any]) -> Any:
        return 1 + eval_expr(self.arg, env)
    def typecheck(self, the_type: Sort, tenv: TypingEnvironment) -> bool:
        """Typing checking for succ(e) from Table 4."""
        if the_type != SInt() and the_type != SNat():
            return False
        return self.arg.typecheck(the_type, tenv)

class Lit(Expression):
    def __init__(self, num: int):
        self.num = num
    def __str__(self) -> str:
        return str(self.num)
    def __repr__(self) -> str:
        return f'Lit({self.num})'
    def eval(self, env: Dict[Variable, Any]) -> Any:
        return self.num
    def typecheck(self, the_type: Sort, tenv: TypingEnvironment) -> bool:
        """Typing checking for numerical literals from Table 4."""
        return (the_type == SNat() and self.num >= 0) or \
                the_type == SInt()

class Choice(Expression):
    """Nondeterministic choice in an expression from section 3, syntax."""
    def __init__(self, e1: Expression, e2: Expression):
        self.e1, self.e2 = e1, e2
    def __str__(self) -> str:
        return f'{self.e1}⊕{self.e2})'
    def __repr__(self) -> str:
        return f'Either({self.e1}, {self.e2})'
    def eval(self, env: Dict[Variable, Any]) -> Any:
        # FIXME What to do here?
        raise NotImplementedError()
    def typecheck(self, the_type: Sort, tenv: TypingEnvironment) -> bool:
        return self.e1.typecheck(the_type, tenv) and \
                self.e2.typecheck(the_type, tenv)

def eval_expr(expr: Expression, env: Dict[Variable, Any]) -> Any:
    if isinstance(expr, int):
        # FIXME Why is this required? We already have Lit?
        return expr
    else:
        return expr.eval(env)

class Process(object):
    def __init__(self) -> None:
        self.environment: Dict[Variable, Any] = {}
    def step(self, role: Participant, state: 'MState') -> Optional['MState']:
        raise NotImplementedError()
    def comm(self, role: Participant, label: Label, data: Any) -> 'Process':
        raise CannotCommunicate()

class MState(object):
    def __init__(self, participants: Dict[Participant, Process]):
        """Initialize state with dict of participants."""
        self.participants = dict(participants)
    def step(self) -> Optional['MState']:
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
    def replace(self, role: Participant, proc: Process) -> 'MState':
        participants1 = dict(self.participants)
        participants1[role] = proc
        return MState(participants1)
    def __repr__(self) -> str:
        strs = [f'\t{role}:\t{str(self.participants[role])}\n'
                for role in self.participants]
        return 'MState(\n{}\t)'.format(''.join(strs))
    def __eq__(self, other: object) -> bool:
        # Prevent use of the default eq implementation.
        # FIXME This will still allow comparison if other implements it,
        #   or if it falls back to identity comparison.
        raise NotImplementedError()

class Inaction(Process):
    def __init__(self) -> None:
        Process.__init__(self)
    def __str__(self) -> str:
        return repr(self)
    def __repr__(self) -> str:
        return f'Inaction({repr(self.environment)})'
    def step(self, role: Participant, state: MState) -> Optional[MState]:
        # Nothing to step
        return None

class CannotCommunicate(Exception):
    pass # Intentionally empty exception.

class Send(Process):
    def __init__(self, destination: Participant, label: Label,
            expr: Expression, continuation: Process):
        Process.__init__(self)
        self.destination, self.label, self.expr, self.continuation = \
                destination, label, expr, continuation
    def __str__(self) -> str:
        return repr(self)
    def __repr__(self) -> str:
        return f'Send({self.destination}, {self.label}, {self.expr}, {self.continuation})'
    def step(self, role: Participant, state0: MState) -> Optional[MState]:
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
    def __init__(self, source: Participant, label: Label, variable: Variable, continuation: Process):
        Process.__init__(self)
        self.source, self.label, self.variable, self.continuation = \
                source, label, variable, continuation
    def __str__(self) -> str:
        return repr(self)
    def __repr__(self) -> str:
        return f'Recv({self.source}, {self.label}, {self.variable}, {self.continuation})'
    def step(self, role: Participant, state: MState) -> None:
        # Waiting for message, cannot step by ourselves.
        return None
    def comm(self, role: Participant, label: Label, data: Any) -> Process:
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

class ExtChoice(Process):
    """An external choice."""
    def __init__(self, *alternatives: Recv):
        Process.__init__(self)
        """The alternatives must be a nonempty list of Recv processes."""
        # FIXME Enforce nonemptyness
        self.alternatives = alternatives
    def __str__(self) -> str:
        return repr(self)
    def __repr__(self) -> str:
        return f'ExtChoice({repr(self.alternatives)})'
    def step(self, role: Participant, state: MState) -> None:
        # We are waiting for another process, cannot step by ourselves.
        return None
    def comm(self, role: Participant, label: Label, data: Any) -> Process:
        # Try each alternative and see if one of them will communicate.
        for proc in self.alternatives:
            try:
                return proc.comm(role, label, data)
            except CannotCommunicate:
                # No, try the other alternatives.
                continue
        # No alternative can communicate.
        raise CannotCommunicate()

def example_2() -> None:
    Bob, Alice, Carol = \
            Participant('Bob'), Participant('Alice'), Participant('Carol')
    l1, l2, l3, l4 = Label(1), Label(2), Label(3), Label(4)
    x = Variable('x')
    PAlice = Send(Bob, l1, Lit(50), Recv(Carol, l3, x, Inaction()))
    PBob = ExtChoice(Recv(Alice, l1, x, Send(Carol, l2, Lit(100), Inaction())),
            Recv(Alice, l4, x, Send(Carol, l2, Lit(2), Inaction())))
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

def section_4_1_example_5() -> None:
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
            Label(1), Label(2), Label(3), Label(4), Label(5)
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

def example_6_1() -> None:
    # First part of example 6
    l1, l2, l3, l4 = Label(1), Label(2), Label(3), Label(4)
    p, q, r = Participant('p'), Participant('q'), Participant('r')
    G1 = GCom(r, q, {l3: (SNat(), GEnd())})
    G2 = GCom(r, q, {l4: (SNat(), GEnd())})
    G = GCom(p, q, {l1: (SNat(), G1), l2: (SBool(), G2)})

    Gp = LInternalChoice(q, {l1: (SNat(), LEnd()), l2: (SBool(), LEnd())})
    Gq = LExternalChoice(p, {
        l1: (SNat(), LExternalChoice(r, {l3: (SNat(), LEnd())})),
        l2: (SBool(), LExternalChoice(r, {l4: (SNat(), LEnd())}))
        })
    if G.project(p) != Gp:
        raise ExampleError((example_6_1, 1))
    if G.project(q) != Gq:
        raise ExampleError((example_6_1, 2))
    if G.project(r) != None:
        raise ExampleError((example_6_1, 3))

def example_6_2() -> None:
    # Second part of example 6
    l1, l2, l3, l4 = Label(1), Label(2), Label(3), Label(4)
    p, q, r = Participant('p'), Participant('q'), Participant('r')
    G1 = GCom(q, r, {l3: (SNat(), GEnd())})
    G2 = GCom(q, r, {l3: (SNat(), GCom(q, r, {l3: (SNat(), GEnd())}))})
    G = GCom(p, q, {l1: (SNat(), G1), l2: (SBool(), G2)})

    Gp = LInternalChoice(q, {l1: (SNat(), LEnd()), l2: (SBool(), LEnd())})
    Gq = LExternalChoice(p, {
        # p?l1(nat).r!l3(nat) 
        l1: (SNat(), LInternalChoice(r, {l3: (SNat(), LEnd())})),
        # p?l2(bool).r!l3(nat).r!l3(nat)
        l2: (SBool(), LInternalChoice(r, {l3: (SNat(), LInternalChoice(r, {l3: (SNat(), LEnd())}))}))
        })
    if G.project(p) != Gp:
        raise ExampleError((example_6_2, 1))
    if G.project(q) != Gq:
        raise ExampleError((example_6_2, 2))
    if G.project(r) != None:
        raise ExampleError((example_6_2, 3))

if __name__ == "__main__":
    example_2()
    example_4()
    section_4_1_example_5()
    example_6_1()
    example_6_2()

