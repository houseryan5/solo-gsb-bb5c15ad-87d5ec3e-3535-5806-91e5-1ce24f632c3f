"""Read-only responsibility-domain (team ownership) audit for a stored plan.

Given a *service -> responsible team* mapping and a per-team simultaneous
on-call cap, the stored dependency graph and stored fault domains are analyzed
(read-only) for:

* **cross-team dependency chains** — every stored dependency edge whose two
  endpoints belong to different teams, both as service-level items
  (:class:`CrossTeamDependency`) and collapsed to teams
  (:class:`TeamDependency`, provider first). Edges touching an unowned
  service cannot be assigned a team pair and are therefore excluded; their
  unowned endpoint is reported among ``unowned_services`` instead;
* **unowned services** — stored-plan services absent from the mapping;
* **team coupling inside fault domains** — for each stored anti-affinity
  domain the distinct teams owning its members; a domain shared by two or
  more teams is *coupled* (their services can never share a release wave, so
  the teams' on-call rotations must coordinate around that domain);
* **coordination order** — the teams connected by cross-team dependency edges
  in dependency-topological order: a provider team (owning prerequisites)
  precedes every consumer team (owning services that depend on them);
  isolated teams need no cross-team coordination and are omitted;
* **per-team footprint and cap violations** — for every team the number of
  stored-plan services it owns and each stored wave in which more than its
  on-call cap of its services release together.

The mapping itself is validated first: assignments of unknown services,
duplicate assignments of one service, assignments without a team and caps
declared for a team that owns nothing raise a structured
:class:`~app.errors.PlanningError` (HTTP 422). Plan services simply left out
of the mapping are *not* an error — they are reported as unowned. The stored
plan is never modified.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass

from .errors import PlanningError
from .schemas import (
    CrossTeamDependency,
    FaultDomainTeamCoupling,
    OwnershipAuditRequest,
    Issue,
    TeamCapViolation,
    TeamDependency,
    TeamOwnership,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_mapping(
    services: list[str], payload: OwnershipAuditRequest
) -> dict[str, str]:
    """Validate the ownership mapping and return service -> team.

    Rejects duplicate service assignments, references to services that are
    not part of the plan, assignments without a team and caps declared for a
    team that does not own any (known) service.
    """
    known = set(services)
    issues: list[Issue] = []

    ownership: dict[str, str] = {}
    seen: set[str] = set()
    for entry in payload.service_owners:
        if entry.service in seen:
            issues.append(
                Issue(
                    code="duplicate_service_owner",
                    message=(
                        f"Service '{entry.service}' is assigned to a team "
                        "more than once; every service has at most one "
                        "responsible team."
                    ),
                    constraint="service_owners",
                    service=entry.service,
                )
            )
            continue
        seen.add(entry.service)
        if entry.service not in known:
            issues.append(
                Issue(
                    code="unknown_service",
                    message=(
                        f"Service '{entry.service}' is not part of this plan "
                        "and cannot appear in its ownership audit."
                    ),
                    constraint="service_owners",
                    service=entry.service,
                )
            )
        elif entry.team is None:
            issues.append(
                Issue(
                    code="missing_service_owner",
                    message=(
                        f"Service '{entry.service}' has no responsible team; "
                        "omit the entry to leave the service unowned instead "
                        "of declaring an empty team."
                    ),
                    constraint="service_owners.service",
                    service=entry.service,
                )
            )
        else:
            ownership[entry.service] = entry.team

    owned_teams = {team for team in ownership.values()}
    for team in sorted(set(payload.team_oncall_caps) - owned_teams):
        issues.append(
            Issue(
                code="unknown_team_cap",
                message=(
                    f"Team '{team}' carries an on-call cap but owns no "
                    "service of this plan."
                ),
                constraint="team_oncall_caps",
                context={"team": team, "cap": payload.team_oncall_caps[team]},
            )
        )

    if issues:
        raise PlanningError(
            "The ownership mapping does not cover the plan consistently: "
            "it contains unknown or duplicate services, ownerless "
            "assignments, or caps for teams without services.",
            issues,
        )

    return ownership


# ---------------------------------------------------------------------------
# Cross-team dependencies and coordination order
# ---------------------------------------------------------------------------


def _cross_team_edges(
    edges: list[tuple[str, str]], ownership: dict[str, str]
) -> list[CrossTeamDependency]:
    """Dependency edges whose two endpoints are owned by different teams.

    An edge with an unowned endpoint cannot be attributed to a team pair and
    is therefore not listed; that endpoint is reported separately as
    unowned by the caller.
    """
    cross: list[CrossTeamDependency] = []
    for service, depends_on in edges:
        service_team = ownership.get(service)
        dependency_team = ownership.get(depends_on)
        if (
            service_team is not None
            and dependency_team is not None
            and service_team != dependency_team
        ):
            cross.append(
                CrossTeamDependency(
                    service=service,
                    service_team=service_team,
                    depends_on=depends_on,
                    dependency_team=dependency_team,
                )
            )
    cross.sort(
        key=lambda e: (
            e.service_team,
            e.dependency_team,
            e.service,
            e.depends_on,
        )
    )
    return cross


def _team_dependencies(
    cross: list[CrossTeamDependency],
) -> list[TeamDependency]:
    """Collapse service-level cross-team edges to provider -> consumer teams."""
    grouped: dict[tuple[str, str], list[CrossTeamDependency]] = defaultdict(list)
    for edge in cross:
        grouped[(edge.dependency_team, edge.service_team)].append(edge)

    result: list[TeamDependency] = []
    for provider, consumer in sorted(grouped):
        edges = sorted(
            grouped[(provider, consumer)],
            key=lambda e: (e.service, e.depends_on),
        )
        result.append(
            TeamDependency(
                provider_team=provider,
                consumer_team=consumer,
                edges=edges,
            )
        )
    return result


def _coordination_order(team_dependencies: list[TeamDependency]) -> list[str]:
    """Kahn topological order over the cross-team dependency graph.

    Direction is provider -> consumer, so providers coordinate before
    consumers. Only teams incident to a cross-team edge take part; ties
    resolve lexicographically (heap) so the order is deterministic.
    """
    providers: dict[str, set[str]] = defaultdict(set)
    consumers: dict[str, set[str]] = defaultdict(set)
    teams: set[str] = set()
    for dep in team_dependencies:
        providers[dep.provider_team].add(dep.consumer_team)
        consumers[dep.consumer_team].add(dep.provider_team)
        teams.add(dep.provider_team)
        teams.add(dep.consumer_team)

    ready = [team for team in teams if not consumers.get(team)]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        team = heapq.heappop(ready)
        order.append(team)
        for consumer in sorted(providers.get(team, ())):
            consumers[consumer].discard(team)
            if not consumers[consumer]:
                heapq.heappush(ready, consumer)
    # The stored service graph is a DAG, so its team quotient is a DAG too.
    assert len(order) == len(teams)
    return order


# ---------------------------------------------------------------------------
# Per-team footprint and on-call cap violations
# ---------------------------------------------------------------------------


def _team_ownership(
    services: list[str],
    ownership: dict[str, str],
    wave_of: dict[str, int],
    caps: dict[str, int],
) -> list[TeamOwnership]:
    """Owned services and cap-exceeding waves per team, sorted by team id."""
    by_team: dict[str, list[str]] = defaultdict(list)
    for service in services:
        team = ownership.get(service)
        if team is not None:
            by_team[team].append(service)

    result: list[TeamOwnership] = []
    for team in sorted(by_team):
        members = sorted(by_team[team])
        cap = caps.get(team)
        violations: list[TeamCapViolation] = []
        if cap is not None:
            in_wave: dict[int, list[str]] = defaultdict(list)
            for service in members:
                in_wave[wave_of[service]].append(service)
            for wave in sorted(in_wave):
                wave_services = sorted(in_wave[wave])
                if len(wave_services) > cap:
                    violations.append(
                        TeamCapViolation(
                            wave=wave,
                            services=wave_services,
                            count=len(wave_services),
                            cap=cap,
                            excess=len(wave_services) - cap,
                        )
                    )
        result.append(
            TeamOwnership(
                team=team,
                service_count=len(members),
                services=members,
                oncall_cap=cap,
                over_limit=violations,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Fault-domain team coupling
# ---------------------------------------------------------------------------


def _fault_domain_coupling(
    domains: list[dict[str, object]], ownership: dict[str, str]
) -> list[FaultDomainTeamCoupling]:
    """Teams sharing each stored anti-affinity domain, in stored domain order."""
    result: list[FaultDomainTeamCoupling] = []
    for domain in domains:
        name = str(domain["domain"])
        members = [str(s) for s in domain["services"]]  # type: list[str]
        services_by_team: dict[str, list[str]] = defaultdict(list)
        unowned: list[str] = []
        for service in members:
            team = ownership.get(service)
            if team is None:
                unowned.append(service)
            else:
                services_by_team[team].append(service)
        teams = sorted(services_by_team)
        result.append(
            FaultDomainTeamCoupling(
                domain=name,
                teams=teams,
                services_by_team={
                    team: sorted(services_by_team[team]) for team in teams
                },
                unowned_services=sorted(unowned),
                coupled=len(teams) >= 2,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnershipAuditOutput:
    teams: list[TeamOwnership]
    unowned_services: list[str]
    cross_team_dependencies: list[CrossTeamDependency]
    team_dependencies: list[TeamDependency]
    coordination_order: list[str]
    fault_domain_coupling: list[FaultDomainTeamCoupling]


def audit_ownership(
    services: list[str],
    edges: list[tuple[str, str]],
    domains: list[dict[str, object]],
    wave_of: dict[str, int],
    payload: OwnershipAuditRequest,
) -> OwnershipAuditOutput:
    """Audit team ownership against the stored graph, domains and waves.

    ``services`` lists every service of the stored plan (sorted), ``edges``
    the dependency edges as ``(service, depends_on)``, ``domains`` the
    stored fault domains and ``wave_of`` each service's 1-based stored wave.
    Nothing is persisted by the caller.
    """
    ownership = _validate_mapping(services, payload)

    unowned = sorted(s for s in services if s not in ownership)

    cross = _cross_team_edges(edges, ownership)
    team_dependencies = _team_dependencies(cross)
    coordination_order = _coordination_order(team_dependencies)

    teams = _team_ownership(
        services, ownership, wave_of, payload.team_oncall_caps
    )
    fault_domain_coupling = _fault_domain_coupling(domains, ownership)

    return OwnershipAuditOutput(
        teams=teams,
        unowned_services=unowned,
        cross_team_dependencies=cross,
        team_dependencies=team_dependencies,
        coordination_order=coordination_order,
        fault_domain_coupling=fault_domain_coupling,
    )
