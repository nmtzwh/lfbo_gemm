from dataclasses import dataclass, field

from .protocol import MeasurementProfile


@dataclass(frozen=True)
class FidelitySchedule:
    """Audited measurement fidelities available to external search policies."""

    macro: MeasurementProfile = field(
        default_factory=lambda: MeasurementProfile(
            profile="macro", warmups=3, samples=3, minimum_sample_ms=100
        )
    )
    micro: MeasurementProfile = field(
        default_factory=lambda: MeasurementProfile(
            profile="micro", warmups=3, samples=5, minimum_sample_ms=200
        )
    )
    finalist: MeasurementProfile = field(
        default_factory=lambda: MeasurementProfile(
            profile="finalist", warmups=3, samples=15, minimum_sample_ms=1000
        )
    )
