"""Pydantic request/response models.

Fault-domain semantics (important): every entry in ``fault_domains`` is an
*anti-affinity* zone. Two services sharing the same zone id must NOT be
released in the same wave, so listing one zone per physical fault domain
expresses "services in this domain cannot co-locate in a wave".
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

# ---- Request models --------------------------------------------------------


class Dependency(BaseModel):
    service: str = Field(..., min_length=1, description="Dependending service")
    depends_on: str = Field(..., min_length=1, description="Required service")

    @field_validator("service", "depends_on")
    @classmethod
    def _strip_and_reject_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("service id must not be blank")
        return v


class FaultDomain(BaseModel):
    domain: str = Field(
        ...,
        min_length=1,
        description="Fault-domain / zone id; members of one domain cannot "
        "share a wave",
    )
    services: list[str] = Field(..., min_length=1)

    @field_validator("domain")
    @classmethod
    def _strip_domain(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("domain id must not be blank")
        return v

    @field_validator("services")
    @classmethod
    def _strip_services(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values]
        if any(not s for s in cleaned):
            raise ValueError("service id must not be blank")
        return cleaned


class PlanCreate(BaseModel):
    name: str | None = Field(
        default=None,
        max_length=200,
        description="Optional human-readable label",
    )
    services: list[str] = Field(..., min_length=1)
    dependencies: list[Dependency] = Field(default_factory=list)
    """service -> depends_on edges; a dependency releases in an earlier wave."""
    max_concurrency: int = Field(..., ge=1, description="Services per wave cap")
    fault_domains: list[FaultDomain] = Field(default_factory=list)
    max_waves: int | None = Field(
        default=None,
        ge=1,
        description="Optional release-window cap; error 422 with reasons if "
        "the optimum needs more waves",
    )

    @field_validator("services", "name")
    @classmethod
    def _strip_service_ids(cls, v: Any) -> Any:
        if isinstance(v, list):
            cleaned = [s.strip() for s in v]
            if any(not s for s in cleaned):
                raise ValueError("service id must not be blank")
            return cleaned
        if isinstance(v, str):
            return v.strip()
        return v


class PlanSimulate(BaseModel):
    """What-if constraint changes applied on top of a stored plan.

    Nothing is persisted. Removals must name constraints that exist in the
    stored plan; additions create new ones. To modify a fault domain, remove
    its id and re-add it with the new member list in the same request. The
    stored ``max_waves`` window is not applied: the simulation reports the
    resulting wave-count delta instead.
    """
    max_concurrency: int | None = Field(
        default=None,
        ge=1,
        description="New per-wave concurrency cap; omit to keep the plan's value",
    )
    add_dependencies: list[Dependency] = Field(default_factory=list)
    remove_dependencies: list[Dependency] = Field(default_factory=list)
    add_fault_domains: list[FaultDomain] = Field(default_factory=list)
    remove_fault_domains: list[str] = Field(
        default_factory=list, description="Ids of fault domains to drop"
    )

    @field_validator("remove_fault_domains")
    @classmethod
    def _strip_domain_ids(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values]
        if any(not v for v in cleaned):
            raise ValueError("domain id must not be blank")
        return cleaned


class GateRequest(BaseModel):
    """Execution progress reported against a stored plan.

    ``completed_services`` have already finished successfully and
    ``failed_services`` have failed in this rollout. The gate decides which
    of the remaining services can still be executed and which are blocked,
    strictly following the stored wave assignment — the plan itself is never
    modified.
    """

    completed_services: list[str] = Field(default_factory=list)
    failed_services: list[str] = Field(default_factory=list)

    @field_validator("completed_services", "failed_services")
    @classmethod
    def _strip_service_ids(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values]
        if any(not s for s in cleaned):
            raise ValueError("service id must not be blank")
        return cleaned


class RollbackPreviewRequest(BaseModel):
    """Services currently live, for which rollback waves are to be previewed.

    The set must be dependency-closed: every prerequisite of a deployed
    service must be deployed as well, otherwise the response is 422 with the
    missing prerequisites named. The stored plan is never modified.
    """

    deployed_services: list[str] = Field(
        ...,
        description=(
            "Services currently live (rolled out). Only these services take "
            "part in the rollback preview; the set must be dependency-closed. "
            "An empty list previews zero rollback waves"
        ),
    )

    @field_validator("deployed_services")
    @classmethod
    def _strip_service_ids(cls, values: list[str]) -> list[str]:
        cleaned = [s.strip() for s in values]
        if any(not s for s in cleaned):
            raise ValueError("service id must not be blank")
        return cleaned


class ServiceTimeline(BaseModel):
    """One service's actual execution window during a rollout."""

    service: str = Field(..., min_length=1, description="Service id of the stored plan")
    started_at: datetime = Field(
        ...,
        description="Actual start instant, ISO 8601 (naive values are read as UTC)",
    )
    finished_at: datetime = Field(
        ...,
        description="Actual finish instant, ISO 8601 (must not precede started_at)",
    )

    @field_validator("service")
    @classmethod
    def _strip_service_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("service id must not be blank")
        return v


class TimelineAuditRequest(BaseModel):
    """Actual per-service execution timeline for a stored plan.

    Exactly one entry per service of the plan is required. Each service must
    have started only after every service of the previous wave and every
    direct dependency had finished; violations are reported per constraint.
    The stored plan is never modified.
    """

    service_timelines: list[ServiceTimeline] = Field(
        ...,
        min_length=1,
        description="One started_at/finished_at pair per service of the plan",
    )


class ServiceOwner(BaseModel):
    """Assignment of one service to the team responsible for it.

    A blank/omitted ``team`` is an incomplete declaration and rejected with a
    structured 422; to express "this service currently has no owner", simply
    leave the service out of the mapping — it is then reported among the
    unowned services. Each plan service may be assigned at most once.
    """

    service: str = Field(..., min_length=1, description="Service id of the stored plan")
    team: str | None = Field(
        default=None,
        description="Responsible team id; null/blank is an incomplete assignment "
        "(omit the entry to leave the service unowned)",
    )

    @field_validator("service")
    @classmethod
    def _strip_service_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("service id must not be blank")
        return v

    @field_validator("team")
    @classmethod
    def _blank_team_is_none(cls, v: str | None) -> str | None:
        # A whitespace-only team name is not a team name: normalize to null
        # so the ownership layer reports one consistent "missing owner" issue.
        if v is None:
            return None
        v = v.strip()
        return v or None


class OwnershipAuditRequest(BaseModel):
    """Service -> responsible-team mapping with per-team on-call caps.

    The mapping may be partial: a plan service left out of ``service_owners``
    is reported as unowned. Entries referencing unknown services, duplicate
    service assignments, entries without a team and caps for teams that own
    nothing are semantic errors (structured 422). Non-positive caps and blank
    service/team ids are schema errors. The stored plan is never modified.
    """

    service_owners: list[ServiceOwner] = Field(
        ...,
        description="Service -> responsible team assignments; each plan "
        "service at most once (uncovered services are reported as unowned)",
    )
    team_oncall_caps: dict[str, Annotated[int, Field(ge=1)]] = Field(
        default_factory=dict,
        description="Maximum number of the team's services that may be "
        "released in the same wave (concurrent on-duty); teams without an "
        "entry are uncapped",
    )

    @field_validator("team_oncall_caps")
    @classmethod
    def _strip_cap_team_ids(
        cls, values: dict[str, int]
    ) -> dict[str, int]:
        cleaned: dict[str, int] = {}
        for team, cap in values.items():
            key = team.strip()
            if not key:
                raise ValueError("team id must not be blank")
            cleaned[key] = cap
        return cleaned


# ---- Response models -------------------------------------------------------


class Wave(BaseModel):
    index: int = Field(..., ge=1, description="1-based wave number")
    services: list[str] = Field(..., description="Sorted service ids in wave")


class Issue(BaseModel):
    """A single, localizable reason that makes planning impossible."""

    code: str
    message: str
    constraint: str
    service: str | None = None
    related_services: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class PlanResult(BaseModel):
    id: int
    name: str | None
    services: list[str]
    waves: list[Wave]
    wave_count: int
    max_concurrency: int
    max_waves: int | None
    dependencies: list[dict[str, str]]
    fault_domains: list[dict[str, Any]]
    optimal: bool
    created_at: str


class PlanSummary(BaseModel):
    id: int
    name: str | None
    wave_count: int
    service_count: int
    max_concurrency: int
    optimal: bool
    created_at: str


class PlanList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PlanSummary]


class PlanDeleted(BaseModel):
    id: int
    deleted: bool = True


class PlanImportResult(PlanResult):
    """A plan restored from an offline bundle via POST /plans/import."""

    sha256: str = Field(
        ..., description="Canonical bundle digest the plan was imported from"
    )
    reused: bool = Field(
        ...,
        description="True when a plan with the same digest already existed "
        "(idempotent replay); false when this call created a new plan",
    )


class ServiceWaveChange(BaseModel):
    service: str
    from_wave: int = Field(..., ge=1, description="Wave in the stored plan (1-based)")
    to_wave: int = Field(..., ge=1, description="Wave in the simulated plan (1-based)")


class PlanSimulation(BaseModel):
    """Outcome of a what-if simulation; never persisted."""

    plan_id: int
    name: str | None
    waves: list[Wave]
    wave_count: int
    original_wave_count: int
    wave_delta: int = Field(
        ..., description="wave_count - original_wave_count (positive: more waves)"
    )
    max_concurrency: int = Field(..., description="Effective per-wave cap used")
    optimal: bool = Field(
        ...,
        description="True when both the wave count and the number of moved "
        "services are proven minimal",
    )
    assignments: list[ServiceWaveChange] = Field(
        ..., description="Every service's old and new wave, sorted by service id"
    )
    moved_services: list[str] = Field(
        ..., description="Services whose wave changed, sorted"
    )
    affected_services: list[str] = Field(
        ...,
        description="Services directly referenced by newly added constraints, "
        "sorted",
    )


class GateBlockedItem(BaseModel):
    """A not-yet-reported service that cannot be executed right now."""

    service: str
    wave: int = Field(..., ge=1, description="Wave of the service in the stored plan")
    reasons: list[Issue] = Field(
        ..., description="Localizable blocking reasons, one per offending dependency"
    )


class GateWaveStatus(BaseModel):
    """Per-wave breakdown of the gate decision (stored plan unchanged)."""

    index: int = Field(..., ge=1)
    executable: list[str] = Field(..., description="Executable members, sorted")
    blocked: list[str] = Field(..., description="Blocked members, sorted")
    completed: list[str] = Field(..., description="Reported-success members, sorted")
    failed: list[str] = Field(..., description="Reported-failed members, sorted")


class GateResult(BaseModel):
    """Outcome of the execution gate for one rollout progress report."""

    plan_id: int
    name: str | None
    wave_count: int = Field(..., description="Stored plan wave count (unchanged)")
    current_wave: int | None = Field(
        ...,
        description="Earliest stored wave that still has an executable service; "
        "null when every service is completed or blocked",
    )
    executable_services: list[str] = Field(
        ...,
        description="Remaining services whose dependencies are all completed, "
        "grouped by wave then sorted by service id",
    )
    blocked_services: list[GateBlockedItem] = Field(
        ...,
        description="Blocked remaining services and their localizable reasons; "
        "empty when nothing is blocked",
    )
    completed_services: list[str] = Field(..., description="Reported successes, sorted")
    failed_services: list[str] = Field(..., description="Reported failures, sorted")
    waves: list[GateWaveStatus] = Field(
        ..., description="Executable/blocked/completed/failed split for every stored wave"
    )


class RollbackPreview(BaseModel):
    """Outcome of a read-only rollback-wave preview; never persisted."""

    plan_id: int
    name: str | None
    deployed_services: list[str] = Field(
        ..., description="Echoed live services, sorted and de-duplicated"
    )
    waves: list[Wave] = Field(
        ...,
        description="Rollback waves (1-based); every service rolls back only "
        "after all live services that depend on it",
    )
    wave_count: int = Field(..., description="Total number of rollback waves")
    max_concurrency: int = Field(
        ..., description="Per-wave cap inherited from the stored plan"
    )
    optimal: bool = Field(
        ..., description="True when the minimum wave count is proven by the solver"
    )


class TimelineViolation(BaseModel):
    """One broken ordering constraint found in the reported execution timeline."""

    service: str = Field(..., description="Service that started too early")
    constraint_type: Literal["previous_wave", "dependency"] = Field(
        ...,
        description="previous_wave: a service of the preceding wave was still "
        "running; dependency: a direct prerequisite was still running",
    )
    constraint: str = Field(
        ...,
        description="Object enforcing the constraint: 'wave N' (1-based) for a "
        "wave barrier, or the id of the direct prerequisite service",
    )
    blocking_service: str | None = Field(
        ...,
        description="The previous-wave/dependency service whose late finish "
        "causes the overlap (null only for an empty preceding wave, which "
        "cannot violate the barrier)",
    )
    overlap_ms: int = Field(
        ...,
        ge=0,
        description="Milliseconds by which service started before the "
        "constraint was satisfied (0 when instants coincide is not a violation)",
    )


class ServiceTiming(BaseModel):
    started_at: datetime
    finished_at: datetime
    duration_ms: int = Field(..., ge=0, description="finished_at - started_at")


class WaveTiming(BaseModel):
    index: int = Field(..., ge=1, description="1-based stored wave number")
    started_at: datetime = Field(..., description="Earliest service start in the wave")
    finished_at: datetime = Field(..., description="Latest service finish in the wave")
    duration_ms: int = Field(..., ge=0, description="finished_at - started_at")
    services: list[str] = Field(..., description="Wave members, sorted")


class CriticalChain(BaseModel):
    """Longest weighted path through the dependency DAG (the actual critical path)."""

    services: list[str] = Field(
        ...,
        description="Services on the chain, prerequisite first; the chain with "
        "the largest summed service duration, ties broken lexicographically"
    )
    duration_ms: int = Field(
        ...,
        ge=0,
        description="Sum of the per-service durations on the chain"
    )


class TimelineAuditResult(BaseModel):
    """Outcome of a read-only execution-timeline audit; never persisted."""

    plan_id: int
    name: str | None
    valid: bool = Field(
        ...,
        description="True when no service started before the previous wave and "
        "every direct dependency had finished",
    )
    violations: list[TimelineViolation] = Field(
        ...,
        description="Every violated wave/dependency constraint, deterministically ordered"
    )
    service_timings: dict[str, ServiceTiming] = Field(
        ..., description="Normalized windows and durations per service id"
    )
    waves: list[WaveTiming] = Field(
        ..., description="Actual per-wave time spans, in stored wave order"
    )
    critical_chain: CriticalChain = Field(
        ...,
        description="Longest total-duration chain on the dependency DAG, "
        "weighted by actual service durations",
    )
    total_duration_ms: int = Field(
        ...,
        ge=0,
        description="Earliest start to latest finish across every service",
    )


# ---- Ownership audit -------------------------------------------------------


class TeamOwnership(BaseModel):
    """One responsible team's footprint against the stored plan."""

    team: str = Field(..., description="Team id")
    service_count: int = Field(
        ..., ge=0, description="Number of stored-plan services owned by the team"
    )
    services: list[str] = Field(
        ..., description="Owned service ids of the stored plan, sorted"
    )
    oncall_cap: int | None = Field(
        ...,
        description="Concurrent per-wave on-call cap reported for the team; "
        "null when no cap applies",
    )
    over_limit: list["TeamCapViolation"] = Field(
        ...,
        description="Stored waves in which more than oncall_cap of the team's "
        "services release together, ordered by wave",
    )


class TeamCapViolation(BaseModel):
    """A stored wave that exceeds a team's simultaneous on-call cap."""

    wave: int = Field(..., ge=1, description="1-based stored wave index")
    services: list[str] = Field(
        ..., description="The team's services placed in this wave, sorted"
    )
    count: int = Field(..., ge=1, description="Number of those services")
    cap: int = Field(..., ge=1, description="Reported per-team concurrent cap")
    excess: int = Field(..., ge=1, description="count - cap")


class CrossTeamDependency(BaseModel):
    """One dependency edge whose two endpoints belong to different teams."""

    service: str = Field(
        ..., description="The depending (downstream) service"
    )
    service_team: str = Field(..., description="Team owning the depending service")
    depends_on: str = Field(..., description="The prerequisite (upstream) service")
    dependency_team: str = Field(..., description="Team owning the prerequisite service")


class TeamDependency(BaseModel):
    """Collapsed cross-team dependency between two teams (provider first)."""

    provider_team: str = Field(
        ...,
        description="Team owning the prerequisite services; must coordinate "
        "before the consumer team",
    )
    consumer_team: str = Field(
        ..., description="Team owning the depending services"
    )
    edges: list[CrossTeamDependency] = Field(
        ...,
        description="Every service-level edge between the two teams, ordered "
        "by consumer then provider service",
    )


class FaultDomainTeamCoupling(BaseModel):
    """Teams coupled because their services share a stored fault domain.

    Members of one anti-affinity domain can never release in the same wave, so
    two teams owning members of the same domain must coordinate (their on-call
    rotations cannot independently choose the same wave).
    """

    domain: str = Field(..., description="Stored fault-domain id")
    teams: list[str] = Field(
        ...,
        description="Distinct teams owning members of the domain, sorted; "
        "unowned members do not appear here",
    )
    services_by_team: dict[str, list[str]] = Field(
        ...,
        description="Domain members per owning team, each list sorted",
    )
    unowned_services: list[str] = Field(
        ..., description="Domain members left without an owner, sorted"
    )
    coupled: bool = Field(
        ...,
        description="True when at least two distinct teams own members of "
        "the domain",
    )


class OwnershipAuditResult(BaseModel):
    """Outcome of a read-only responsibility-domain audit; never persisted."""

    plan_id: int
    name: str | None
    teams: list[TeamOwnership] = Field(
        ...,
        description="One row per team named in the mapping, sorted by team id; "
        "each lists the owned services and every cap-exceeding wave",
    )
    unowned_services: list[str] = Field(
        ...,
        description="Plan services absent from the service_owners mapping, sorted"
    )
    cross_team_dependencies: list[CrossTeamDependency] = Field(
        ...,
        description="Dependency edges joining different teams, ordered by "
        "consumer service then prerequisite service; edges touching an "
        "unowned service are excluded (the unowned endpoint is reported "
        "separately in unowned_services)",
    )
    team_dependencies: list[TeamDependency] = Field(
        ...,
        description="The same cross-team edges collapsed to teams (provider "
        "first), ordered by provider then consumer",
    )
    coordination_order: list[str] = Field(
        ...,
        description="Teams that must coordinate, in dependency-topological "
        "order: a provider team precedes every consumer team; isolated teams "
        "(no cross-team edges) are omitted; ties resolved lexicographically",
    )
    fault_domain_coupling: list[FaultDomainTeamCoupling] = Field(
        ...,
        description="Per stored fault domain the teams sharing it, in stored "
        "domain order; domains with at least two teams are coupled",
    )


# ---- Service slack report --------------------------------------------------


class ServiceSlack(BaseModel):
    """One service's wave-level slack within the stored minimum-wave plan."""

    service: str = Field(..., description="Service id of the stored plan")
    wave: int = Field(
        ..., ge=1, description="Stored wave of the service (1-based)"
    )
    latest_wave: int = Field(
        ...,
        ge=1,
        description="Latest wave the service can move to without adding a "
        "wave or violating any dependency, concurrency or fault-domain "
        "constraint (1-based)",
    )
    slack_waves: int = Field(
        ...,
        ge=0,
        description="Maximum number of waves the service can move backwards "
        "(later): latest_wave - wave; 0 means the service is critical",
    )


class CriticalService(BaseModel):
    """A zero-slack service and the critical downstream it constrains."""

    service: str = Field(..., description="Zero-slack (critical) service id")
    wave: int = Field(..., ge=1, description="Stored wave of the service (1-based)")
    constrained_downstream: list[str] = Field(
        ...,
        description="Zero-slack services reachable from this service over "
        "dependency edges that are tight (exactly one stored wave apart), "
        "i.e. the critical-chain continuation constrained by it; sorted",
    )


class SlackReport(BaseModel):
    """Read-only per-service slack report; never persisted."""

    plan_id: int
    name: str | None
    wave_count: int = Field(
        ..., description="Stored plan wave count; the report never adds a wave"
    )
    optimal: bool = Field(
        ...,
        description="True when every per-service latest-wave maximum was "
        "proven optimal by the solver; False after a time limit means some "
        "slack_waves values may be conservative lower bounds",
    )
    services: list[ServiceSlack] = Field(
        ..., description="One slack row per service, sorted by service id"
    )
    critical_services: list[CriticalService] = Field(
        ...,
        description="Services with slack_waves == 0, with the zero-slack "
        "downstream constrained by each; sorted by service id",
    )


# ---- Error envelope --------------------------------------------------------


class ErrorBody(BaseModel):
    error: Literal["planning_failed", "not_found", "validation_error", "internal_error"]
    message: str
    issues: list[Issue] = Field(default_factory=list)
