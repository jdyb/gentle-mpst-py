#!/usr/bin/env python3

"""
Notes to reader:
    * Methods of subclasses are more likely to be documented.
    * All methods have type annotations, even when those are redudant,
      for simpler reading.
"""

from typing import Dict, Tuple, Set

# Section 3 "Synchronous Multiparty Session Calculus", Notation 01 (Base Sets)

class Label(object):
    def __init__(self, name):
        self.lname = name
    def __str__(self):
        return str(self.lname)

class Participant(object):
    """Session participant, from "Notation 01 (Base sets)" """
    def __init__(self, name):
        self.rname = name
    def __str__(self):
        return self.rname

# FIXME From where are the sorts

class Sort(object):
    """Sorts and global types"""
    pass

class Nat(Sort):
    pass

class Int(Sort):
    pass

class Bool(Sort):
    pass

# The subclasses of Local are from section "4.1 Types and Projectsions" and
# "Definition 3 (Local Session Types)"

class LocalT(object):
    """Local type."""
    def pt(self) -> Set[Participant]:
        raise NotImplementedError()

class LEnd(LocalT):
    """Local termination"""
    def pt(self) -> Set[Participant]:
        return set()

class LExternalChoice(LocalT):
    """Local Type for External Choice"""
    def __init__(self, p, alternatives: Dict[Label, Tuple[Sort,LocalT]]):
        self.p, self.alternatives = p, alternatives
    def pt(self) -> Set[Participant]:
        pts = set((self.p,))
        for label in self.alternatives:
            sort, ltype = self.alternatives[label]
            pts.add(ltype.pt())
        return pts

class LInternalChoice(LocalT):
    """Local Type for Internal Choice"""
    def __init__(self, q, alternatives: Dict[Label, Tuple[Sort,LocalT]]):
        self.q, self.alternatives = q, alternatives
    def pt(self) -> Set[Participant]:
        pts = set((self.q,))
        for label in self.alternatives:
            sort, ltype = self.alternatives[label]
            pts.add(ltype.pt())
        return pts

class LVariable(LocalT):
    def __init__(self, name: str):
        self.ltvname = name
    def pt(self) -> Set[Participant]:
        return set()

class LRec(LocalT):
    def __init__(self, ltvariable: LVariable, local_type: LocalT):
        self.ltvariable, self.local_type = ltvariable, local_type
    def pt(self) -> Set[Participant]:
        return self.local_type.pt()


# The subclasses of GlobalT are from section "4.1 Types and Projections"
# definition 2. 

class GlobalT(object):
    """Global type"""
    def pt(self):
        """Compute the set of participants of a global type.
        Returns the set of participants."""
        raise NotImplementedError()
    def simp_projection(self, r: Participant) -> LocalT:
        """Simple (no merging) projection. See 'Definition 4'."""
        raise NotImplementedError()

class GEnd(GlobalT):
    """Global type signifying terminated protocol."""
    def pt(self):
        return set()
    def simp_projection(self, r: Participant) -> LocalT:
        pass

class GTVar(GlobalT):
    """Global type variable"""
    def __init__(self, name):
        self.gtvname = name
    def pt(self):
        return set()

class GRec(GlobalT):
    """Recursive global type"""
    def __init__(self, gtvariable: GTVar, global_type: GlobalT):
        self.gtvariable, self.global_type = gtvariable, global_type
    def pt(self):
        return self.global_type.pt()

class GCom(GlobalT):
    """Global type for message communication between two participants."""
    def __init__(self, source, destination, alternatives):
        """Alternatives is a dict, label keys."""
        self.source, self.destination, self.alternatives = \
                source, destination, alternatives
    def pt(self):
        pts = set((self.source, self.destination))
        for label in self.alternatives:
            Si, Gi = self.alternatives[label]
            pts.add(Gi.pt())
        return pts

class Expression(object):
    pass

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
    pass

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
        proc_dst = state.participants[self.destination]
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

Bob, Alice, Carol = Participant('Bob'), Participant('Alice'), Participant('Carol')
l1, l2, l3, l4 = Label(1), Label(2), Label(3), Label(4)
x = Variable('x')

PAlice = Send(Bob, l1, 50, Recv(Carol, l3, x, Inaction()))
PBob = ExtChoice(Recv(Alice, l1, x, Send(Carol, l2, 100, Inaction())),
        Recv(Alice, l4, x, Send(Carol, l2, 2, Inaction())))
PCarol = Recv(Bob, l2, x, Send(Alice, l3, Succ(x), Inaction()))

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
                print(f'# Stepped {role}')
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

state = MState({Alice: PAlice, Bob: PBob, Carol: PCarol})

while state:
    print(repr(state))
    state = state.step()

