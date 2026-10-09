"""Dynamic environment SPEC: node types, node instances and the axiom set.

Everything that describes a concrete environment lives in a JSON spec file:
the node-type schema, the axiom set and the initial nodes/objectives. The engine
(``engine.py``) never branches on a spec id, a node-type name, a tag value or a
primitive name; it only interprets the DECLARED schema and axioms.

A spec has the following shape (all keys are data, none is engine knowledge)::

    {
      "spec_id": "...",
      "description": "...",
      "node_types": {
         "<type>": {
            "semantics": "literal" | "combination",
            "role": "value" | "tag" | "objective" | "plain",
            "value_field": "<field>",              # literal types only
            "fields": [{"name": "...", "type": "int|str|bool|node|nodes|any"}]
         }
      },
      "combine_axioms": [
         {"id": "...", "tag": <int|str>, "arity": 2, "op": "sum",
          "params": {...}, "active": true}
      ],
      "build_axioms": [
         {"id": "...", "tag_node": "<node-id>", "arity": 2,
          "node_type": "grp", "operand_type": null, "active": true}
      ],
      "dynamic_axioms": [
         {"id": "...", "when": <condition>, "effect": [<effect>, ...],
          "active": true, "once": true}
      ],
      "nodes": {"<id>": {"type": "<type>", "<field>": <value>}},
      "objectives": {"<objective-node-id>": 0|1},
      "guards": {"<objective-node-id>": <condition>},
      "dynamic_node_type": "grp" | null,
      "max_combine_arity": 2,
      "step_cost": 0.01,
      "goal_reward": 1.0,
      "max_steps": 12
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Directory that ships the spec fixtures, relative to the repository root.
SPEC_DIR = os.path.join("data", "dynamic_env")

VALID_FIELD_TYPES = ("int", "str", "bool", "node", "nodes", "any")
VALID_SEMANTICS = ("literal", "combination")
VALID_ROLES = ("value", "tag", "objective", "plain")
VALID_OPS = (
    "sum", "sub", "mul", "div", "mod", "neg", "succ",
    "eq", "lt", "gt", "le", "ge",
    "and", "or", "not", "if",
    "identity", "constant", "list", "head", "tail", "len",
    "min", "max", "cat",
)
VALID_CONDITIONS = (
    "always", "objective_is", "all_objectives_match", "exists_value",
    "node_type_count", "count_nodes", "axiom_active", "and", "or", "not",
)
VALID_EFFECTS = (
    "activate_axiom", "deactivate_axiom", "add_node", "remove_node",
    "set_objective", "rewrite_node",
)
VALID_CMPS = ("eq", "ne", "lt", "le", "gt", "ge")


class SpecError(Exception):
    """A spec is malformed or internally inconsistent."""


def _as_tuple(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: str

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "FieldSpec":
        name = raw.get("name")
        ftype = raw.get("type", "any")
        if not isinstance(name, str) or not name:
            raise SpecError("field needs a non-empty 'name': %r" % (raw,))
        if ftype not in VALID_FIELD_TYPES:
            raise SpecError("field %r has unknown type %r" % (name, ftype))
        return FieldSpec(name=name, type=ftype)


@dataclass(frozen=True)
class NodeType:
    name: str
    fields: Tuple[FieldSpec, ...]
    semantics: str
    role: str
    value_field: Optional[str]

    @staticmethod
    def from_dict(name: str, raw: Mapping[str, Any]) -> "NodeType":
        semantics = raw.get("semantics", "literal")
        role = raw.get("role", "plain")
        if semantics not in VALID_SEMANTICS:
            raise SpecError("node type %r: unknown semantics %r" % (name, semantics))
        if role not in VALID_ROLES:
            raise SpecError("node type %r: unknown role %r" % (name, role))
        fields = tuple(FieldSpec.from_dict(f) for f in raw.get("fields", ()))
        value_field = raw.get("value_field")
        if semantics == "literal":
            if not value_field:
                raise SpecError("literal node type %r needs 'value_field'" % name)
            if value_field not in [f.name for f in fields]:
                raise SpecError(
                    "node type %r value_field %r is not a declared field"
                    % (name, value_field)
                )
        return NodeType(
            name=name, fields=fields, semantics=semantics, role=role,
            value_field=value_field,
        )

    def field(self, name: str) -> Optional[FieldSpec]:
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass(frozen=True)
class Node:
    """A node instance: a declared type plus field values, keyed by a local id."""

    id: str
    type: str
    fields: Tuple[Tuple[str, Any], ...] = ()

    def get(self, name: str, default: Any = None) -> Any:
        for key, value in self.fields:
            if key == name:
                return value
        return default

    def canonical(self) -> str:
        body = ",".join("%s=%r" % (k, v) for k, v in self.fields)
        return "%s#%s(%s)" % (self.id, self.type, body)

    @staticmethod
    def from_dict(node_id: str, raw: Mapping[str, Any]) -> "Node":
        ntype = raw.get("type")
        if not isinstance(ntype, str) or not ntype:
            raise SpecError("node %r needs a 'type': %r" % (node_id, raw))
        fields = tuple(
            (str(k), v) for k, v in raw.items() if k != "type"
        )
        return Node(id=node_id, type=ntype, fields=fields)


@dataclass(frozen=True)
class CombineAxiom:
    id: str
    tag: Any
    arity: Optional[int]
    op: str
    params: Mapping[str, Any]
    active: bool


@dataclass(frozen=True)
class BuildAxiom:
    id: str
    tag_node: str
    arity: int
    node_type: str
    operand_type: Optional[str]
    active: bool


@dataclass(frozen=True)
class DynamicAxiom:
    id: str
    when: Mapping[str, Any]
    effect: Tuple[Mapping[str, Any], ...]
    active: bool
    once: bool


@dataclass(frozen=True)
class Spec:
    spec_id: str
    description: str
    node_types: Mapping[str, NodeType]
    combine_axioms: Tuple[CombineAxiom, ...]
    build_axioms: Tuple[BuildAxiom, ...]
    dynamic_axioms: Tuple[DynamicAxiom, ...]
    nodes: Mapping[str, Node]
    objectives: Mapping[str, int]
    guards: Mapping[str, Mapping[str, Any]]
    dynamic_node_type: Optional[str]
    max_combine_arity: int
    step_cost: float
    goal_reward: float
    max_steps: int
    _combine_index: Mapping[Tuple[Any, int], CombineAxiom] = field(
        default_factory=dict, repr=False, compare=False
    )
    _axioms_by_id: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )

    def combine_axiom(self, tag: Any, arity: int) -> Optional[CombineAxiom]:
        return (
            self._combine_index.get((tag, arity))
            or self._combine_index.get((tag, None))
        )

    def axiom(self, axiom_id: str) -> Optional[Any]:
        return self._axioms_by_id.get(axiom_id)

    def initial_active(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                ax.id
                for ax in (
                    tuple(self.combine_axioms)
                    + tuple(self.build_axioms)
                    + tuple(self.dynamic_axioms)
                )
                if ax.active
            )
        )


def _validate_condition(cond: Mapping[str, Any], where: str) -> None:
    if not isinstance(cond, Mapping):
        raise SpecError("%s: condition must be an object, got %r" % (where, cond))
    op = cond.get("op")
    if op not in VALID_CONDITIONS:
        raise SpecError("%s: unknown condition op %r" % (where, op))
    if op in ("and", "or"):
        args = cond.get("args")
        if not isinstance(args, list) or not args:
            raise SpecError("%s: '%s' needs a non-empty 'args' list" % (where, op))
        for i, arg in enumerate(args):
            _validate_condition(arg, "%s.%s[%d]" % (where, op, i))
    elif op == "not":
        _validate_condition(cond.get("arg"), "%s.not" % where)
    elif op in ("exists_value", "node_type_count", "count_nodes"):
        cmp_ = cond.get("cmp")
        if cmp_ not in VALID_CMPS:
            raise SpecError("%s: unknown cmp %r" % (where, cmp_))
        if "value" not in cond:
            raise SpecError("%s: '%s' needs 'value'" % (where, op))
    elif op == "objective_is":
        if not isinstance(cond.get("id"), str) or "value" not in cond:
            raise SpecError("%s: 'objective_is' needs 'id' and 'value'" % where)
    elif op == "axiom_active":
        if not isinstance(cond.get("id"), str):
            raise SpecError("%s: 'axiom_active' needs 'id'" % where)


def _validate_effect(effect: Mapping[str, Any], where: str) -> None:
    if not isinstance(effect, Mapping):
        raise SpecError("%s: effect must be an object, got %r" % (where, effect))
    op = effect.get("op")
    if op not in VALID_EFFECTS:
        raise SpecError("%s: unknown effect op %r" % (where, op))
    if op in ("activate_axiom", "deactivate_axiom"):
        if not isinstance(effect.get("id"), str):
            raise SpecError("%s: '%s' needs 'id'" % (where, op))
    elif op in ("add_node", "rewrite_node"):
        if not isinstance(effect.get("id"), str):
            raise SpecError("%s: '%s' needs 'id'" % (where, op))
        if not isinstance(effect.get("node"), Mapping):
            raise SpecError("%s: '%s' needs a 'node' object" % (where, op))
    elif op == "remove_node":
        if not isinstance(effect.get("id"), str):
            raise SpecError("%s: 'remove_node' needs 'id'" % where)
    elif op == "set_objective":
        if not isinstance(effect.get("id"), str) or effect.get("value") not in (0, 1):
            raise SpecError("%s: 'set_objective' needs 'id' and 0/1 'value'" % where)


def _validate_node_against_type(node: Node, ntype: NodeType, where: str) -> None:
    declared = set()
    for f in ntype.fields:
        declared.add(f.name)
        if node.get(f.name) is None:
            raise SpecError("%s: node %r misses field %r" % (where, node.id, f.name))
    extra = set(k for k, _ in node.fields) - declared
    if extra:
        raise SpecError("%s: node %r has undeclared fields %s"
                        % (where, node.id, sorted(extra)))


def spec_from_dict(raw: Mapping[str, Any]) -> Spec:
    """Parse and validate a spec dict; raise :class:`SpecError` on any defect."""
    if not isinstance(raw, Mapping):
        raise SpecError("spec must be a JSON object")
    spec_id = raw.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id:
        raise SpecError("spec needs a non-empty 'spec_id'")

    raw_types = raw.get("node_types")
    if not isinstance(raw_types, Mapping) or not raw_types:
        raise SpecError("spec %r needs a non-empty 'node_types'" % spec_id)
    node_types: Dict[str, NodeType] = {}
    for name, body in raw_types.items():
        node_types[name] = NodeType.from_dict(name, body)

    nodes: Dict[str, Node] = {}
    for node_id, body in (raw.get("nodes") or {}).items():
        node = Node.from_dict(node_id, body)
        if node.type not in node_types:
            raise SpecError("node %r has unknown type %r" % (node_id, node.type))
        _validate_node_against_type(node, node_types[node.type], spec_id)
        nodes[node_id] = node

    combine_axioms: List[CombineAxiom] = []
    for body in raw.get("combine_axioms") or ():
        op = body.get("op")
        if op not in VALID_OPS:
            raise SpecError("combine axiom %r: unknown op %r" % (body.get("id"), op))
        arity = body.get("arity")
        if arity is not None and (not isinstance(arity, int) or arity < 0):
            raise SpecError("combine axiom %r: bad arity %r" % (body.get("id"), arity))
        combine_axioms.append(CombineAxiom(
            id=str(body.get("id")), tag=body.get("tag"), arity=arity, op=op,
            params=dict(body.get("params") or {}), active=bool(body.get("active", True)),
        ))

    build_axioms: List[BuildAxiom] = []
    for body in raw.get("build_axioms") or ():
        node_type = body.get("node_type")
        if node_type not in node_types:
            raise SpecError("build axiom %r: unknown node_type %r"
                            % (body.get("id"), node_type))
        if node_types[node_type].semantics != "combination":
            raise SpecError("build axiom %r: node_type %r is not a combination type"
                            % (body.get("id"), node_type))
        tag_node = body.get("tag_node")
        if tag_node not in nodes:
            raise SpecError("build axiom %r: unknown tag_node %r"
                            % (body.get("id"), tag_node))
        arity = body.get("arity")
        if not isinstance(arity, int) or arity < 0:
            raise SpecError("build axiom %r: bad arity %r" % (body.get("id"), arity))
        operand_type = body.get("operand_type")
        if operand_type is not None and operand_type not in node_types:
            raise SpecError("build axiom %r: unknown operand_type %r"
                            % (body.get("id"), operand_type))
        build_axioms.append(BuildAxiom(
            id=str(body.get("id")), tag_node=str(tag_node), arity=arity,
            node_type=str(node_type), operand_type=operand_type,
            active=bool(body.get("active", True)),
        ))

    dynamic_axioms: List[DynamicAxiom] = []
    for body in raw.get("dynamic_axioms") or ():
        _validate_condition(body.get("when") or {"op": "always"},
                            "dynamic axiom %r.when" % body.get("id"))
        effect = tuple(body.get("effect") or ())
        for i, eff in enumerate(effect):
            _validate_effect(eff, "dynamic axiom %r.effect[%d]" % (body.get("id"), i))
        dynamic_axioms.append(DynamicAxiom(
            id=str(body.get("id")),
            when=dict(body.get("when") or {"op": "always"}),
            effect=effect,
            active=bool(body.get("active", True)),
            once=bool(body.get("once", True)),
        ))

    objectives: Dict[str, int] = {}
    for oid, target in (raw.get("objectives") or {}).items():
        if oid not in nodes:
            raise SpecError("objective %r is not a declared node" % oid)
        if node_types[nodes[oid].type].role != "objective":
            raise SpecError("objective %r has a non-objective node type %r"
                            % (oid, nodes[oid].type))
        ntype = node_types[nodes[oid].type]
        initial = nodes[oid].get(ntype.value_field)
        if initial not in (0, 1):
            raise SpecError(
                "objective %r initial value must be 0 or 1, got %r" % (oid, initial)
            )
        if target not in (0, 1):
            raise SpecError("objective %r target must be 0 or 1, got %r" % (oid, target))
        objectives[oid] = int(target)
    if not objectives:
        raise SpecError("spec %r declares no objectives" % spec_id)

    guards: Dict[str, Mapping[str, Any]] = {}
    for oid, cond in (raw.get("guards") or {}).items():
        if oid not in objectives:
            raise SpecError("guard for %r which is not an objective" % oid)
        _validate_condition(cond, "guard[%s]" % oid)
        guards[oid] = dict(cond)

    dynamic_node_type = raw.get("dynamic_node_type")
    if dynamic_node_type is not None:
        if dynamic_node_type not in node_types:
            raise SpecError("dynamic_node_type %r is unknown" % dynamic_node_type)
        if node_types[dynamic_node_type].semantics != "combination":
            raise SpecError("dynamic_node_type %r is not a combination type"
                            % dynamic_node_type)

    max_combine_arity = int(raw.get("max_combine_arity", 2))
    if max_combine_arity < 0:
        raise SpecError("max_combine_arity must be >= 0")

    combine_index: Dict[Tuple[Any, Optional[int]], CombineAxiom] = {}
    for ax in combine_axioms:
        # An exact (tag, arity) entry always wins; an arity-less axiom is also
        # registered under (tag, None) and matched as the fallback for any arity.
        combine_index[(ax.tag, ax.arity)] = ax
    axioms_by_id: Dict[str, Any] = {}
    for ax in tuple(combine_axioms) + tuple(build_axioms) + tuple(dynamic_axioms):
        if ax.id in axioms_by_id:
            raise SpecError("duplicate axiom id %r" % ax.id)
        axioms_by_id[ax.id] = ax

    return Spec(
        spec_id=spec_id,
        description=str(raw.get("description", "")),
        node_types=node_types,
        combine_axioms=tuple(combine_axioms),
        build_axioms=tuple(build_axioms),
        dynamic_axioms=tuple(dynamic_axioms),
        nodes=nodes,
        objectives=objectives,
        guards=guards,
        dynamic_node_type=dynamic_node_type,
        max_combine_arity=max_combine_arity,
        step_cost=float(raw.get("step_cost", 0.01)),
        goal_reward=float(raw.get("goal_reward", 1.0)),
        max_steps=int(raw.get("max_steps", 16)),
        _combine_index=combine_index,
        _axioms_by_id=axioms_by_id,
    )


def load_spec(path: str) -> Spec:
    """Load a spec from a JSON file (the ONLY way an environment is built)."""
    if not os.path.isfile(path):
        raise SpecError("spec file not found: %s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return spec_from_dict(raw)


def load_spec_by_id(spec_id: str, spec_dir: str = SPEC_DIR) -> Spec:
    """Load one of the shipped fixtures by its file stem."""
    return load_spec(os.path.join(spec_dir, spec_id + ".json"))


def list_spec_files(spec_dir: str = SPEC_DIR) -> List[str]:
    if not os.path.isdir(spec_dir):
        return []
    return sorted(
        os.path.join(spec_dir, name)
        for name in os.listdir(spec_dir)
        if name.endswith(".json")
    )
