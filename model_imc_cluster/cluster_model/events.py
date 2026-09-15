"""Small dependency/resource timeline. Durations are assumptions, not chip PPA."""
from dataclasses import dataclass, field
import math
from .digital import ModelError


@dataclass
class Event:
    """功能事件；dependencies 是本时间表内的事件 ID，时间不是已标定周期。"""
    id: int
    name: str
    time: float
    end: float
    resource: str
    context: str
    dependencies: tuple[int, ...] = ()
    details: dict = field(default_factory=dict)


class EventSchedule:
    """记录依赖和资源占用，校验已声明的约束，不替控制器自动调度。"""
    def __init__(self, context="default"):
        self.context = context
        self.events = []

    def schedule(self, name, start, duration=0.0, *, resource="controller", dependencies=(), **details):
        if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration < 0:
            raise ValueError("invalid event time/duration")
        for dep in dependencies:
            if not 0 <= dep < len(self.events):
                raise ModelError("missing_dependency", event=name, dependency=dep)
            if self.events[dep].end > start:
                raise ModelError("dependency_not_ready", event=name, dependency=dep, time=start)
        for event in self.events:
            if event.resource == resource and duration > 0 and event.end > start and event.time < start + duration:
                raise ModelError("resource_conflict", resource=resource, event=name, previous=event.id)
        # ID 按加入顺序稳定编号；零时长数字事件用依赖表达顺序，而非吞吐率。
        event = Event(len(self.events), name, float(start), float(start + duration), resource,
                      self.context, tuple(dependencies), details)
        self.events.append(event)
        return event.id


@dataclass(frozen=True)
class Timing:
    """Offsets from candidate_ready; explicit switch times are never repaired."""
    share_at: float | None = None
    compare_at: float | None = None

    def __post_init__(self):
        for value in (self.share_at, self.compare_at):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("timing offsets must be nonnegative and finite")
