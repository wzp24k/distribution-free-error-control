"""Compile expert Boolean marker definitions into evaluable predicates.

The depot's CellTypes tables define each population by a Boolean expression over
protein channels, e.g. "CD4&CD3", "(CD56|CD57)&CD45", "CD38&!IgA". Those
definitions are expert knowledge and give a biological consistency check that is
independent of any classifier.

Grammar:  expr   := term ("|" term)*
          term   := factor ("&" factor)*
          factor := "!" factor | "(" expr ")" | IDENT
"""

import re

import numpy as np


class LogicError(ValueError):
    pass


_TOKEN = re.compile(r"\s*(?:(\()|(\))|(!)|(&)|(\|)|([A-Za-z0-9_\-\.]+))")


def tokenize(text):
    pos, out = 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise LogicError("cannot tokenize at position %d: %r" % (pos, text[pos:pos + 20]))
        pos = m.end()
        if m.group(1):
            out.append(("(", "("))
        elif m.group(2):
            out.append((")", ")"))
        elif m.group(3):
            out.append(("not", "!"))
        elif m.group(4):
            out.append(("and", "&"))
        elif m.group(5):
            out.append(("or", "|"))
        else:
            out.append(("ident", m.group(6)))
    return out


class Node:
    def idents(self):
        raise NotImplementedError

    def evaluate(self, values):
        raise NotImplementedError


class Ident(Node):
    def __init__(self, name):
        self.name = name

    def idents(self):
        return {self.name}

    def evaluate(self, values):
        if self.name not in values:
            raise LogicError("channel %r not available" % self.name)
        return values[self.name]

    def __repr__(self):
        return self.name


class Not(Node):
    def __init__(self, child):
        self.child = child

    def idents(self):
        return self.child.idents()

    def evaluate(self, values):
        return ~self.child.evaluate(values)

    def __repr__(self):
        return "!%r" % (self.child,)


class Bin(Node):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right

    def idents(self):
        return self.left.idents() | self.right.idents()

    def evaluate(self, values):
        a = self.left.evaluate(values)
        b = self.right.evaluate(values)
        if self.op == "and":
            return np.logical_and(a, b)
        return np.logical_or(a, b)

    def __repr__(self):
        return "(%r %s %r)" % (self.left, self.op, self.right)


def parse(text):
    tokens = tokenize(text)
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else (None, None)

    def take(kind=None):
        k, v = peek()
        if kind is not None and k != kind:
            raise LogicError("expected %s, got %s (%r) in %r" % (kind, k, v, text))
        pos[0] += 1
        return k, v

    def parse_or():
        node = parse_and()
        while peek()[0] == "or":
            take("or")
            node = Bin("or", node, parse_and())
        return node

    def parse_and():
        node = parse_not()
        while peek()[0] == "and":
            take("and")
            node = Bin("and", node, parse_not())
        return node

    def parse_not():
        if peek()[0] == "not":
            take("not")
            return Not(parse_not())
        return parse_atom()

    def parse_atom():
        k, v = peek()
        if k == "(":
            take("(")
            node = parse_or()
            take(")")
            return node
        if k == "ident":
            take("ident")
            return Ident(v)
        raise LogicError("unexpected token %s (%r) in %r" % (k, v, text))

    node = parse_or()
    if pos[0] != len(tokens):
        raise LogicError("trailing tokens in %r at %r" % (text, tokens[pos[0]:]))
    return node


# Channel names differ between panels; map them onto a common key so that a rule
# written for one dataset can be evaluated on another panel.
_SYNONYMS = {
    "pan_keratin": "Keratin",
    "cytokeratin": "Keratin",
    "keratin": "Keratin",
    "ecad": "ECAD",
    "cd326": "ECAD",
    "hladr": "HLADRDPDQ",
    "hladrdpdq": "HLADRDPDQ",
    "hla1": "HLADRDPDQ",
    "nakatpase": "NAKATPASE",
    "nak_atpase": "NAKATPASE",
    "fe56": "Fe",
    "fe": "Fe",
    "collagen": "Collagen",
    "dna": "DNA",
    "dsdna": "DNA",
    "cd16": "CD16",
    "cd11b": "CD11b",
}


def canonical(channel):
    c = str(channel).strip()
    return _SYNONYMS.get(c.lower(), c)


def _canonicalised(node):
    """Rewrite every identifier node onto the shared channel vocabulary."""
    if isinstance(node, Ident):
        return Ident(canonical(node.name))
    if isinstance(node, Not):
        return Not(_canonicalised(node.child))
    if isinstance(node, Bin):
        return Bin(node.op, _canonicalised(node.left), _canonicalised(node.right))
    return node


class CompiledRule:
    """A parsed marker definition plus its provenance.

    Identifiers are canonicalised once, at compile time, so that a rule written
    for one dataset's panel can be evaluated against another panel's channels --
    without this, ``idents`` and the expression disagree about the channel names.
    """

    def __init__(self, cell_type, text):
        self.cell_type = cell_type
        self.text = text
        self.ast = _canonicalised(parse(text))
        self.idents = set(self.ast.idents())

    def evaluate(self, values):
        return self.ast.evaluate(values)

    def __repr__(self):
        return "CompiledRule(%r, %r)" % (self.cell_type, self.text)


def compile_rules(frame, cell_col="CellType", marker_col="PrimaryMarker",
                  skip=("Unidentified",)):
    """Build {cell type -> CompiledRule} from a CellTypes table."""
    rules, failed = {}, []
    for _, row in frame.iterrows():
        ct = str(row[cell_col])
        raw = row.get(marker_col) if hasattr(row, "get") else row[marker_col]
        if ct in skip or raw is None or raw != raw:
            continue
        text = str(raw).strip()
        if not text or text.lower() == "nan":
            continue
        try:
            rules[ct] = CompiledRule(ct, text)
        except LogicError as exc:
            failed.append((ct, text, str(exc)))
    return rules, failed

